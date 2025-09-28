#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![feature(allocator_api)]
#![feature(iter_array_chunks)]

use std::alloc::Allocator;
use std::array::from_ref;

use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::algorithms::fft::cooley_tuckey::bitreverse;
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::divisibility::*;
use feanor_math::group::AbelianGroupStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, BigIntRing, IntegerRingStore};
use feanor_math::primitive_int::*;
use feanor_math::seq::permute::permute;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;

use feanor_math::rings::extension::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::{VectorFn, VectorView};
use fheanor::bfv::{BFVInstantiation, Ciphertext, CiphertextRing, PlaintextRing, Pow2BFV};
use fheanor::bgv::SecretKeyDistribution;
use fheanor::ciphertext_ring::indices::RNSFactorIndexList;
use fheanor::ciphertext_ring::BGFVCiphertextRing;
use fheanor::circuit::*;
use fheanor::clpx::encoding::CLPXPlaintextRing;
use fheanor::clpx::{CLPXInstantiation, Pow2CLPX};
use fheanor::digit_extract::polys::{bounded_digit_retain_poly, poly_to_circuit};
use fheanor::filename_keys;
use fheanor::NiceZn;
use fheanor::gadget_product::digits::RNSGadgetVectorDigitIndices;
use fheanor::lin_transform::pow2;
use fheanor::number_ring::galois::CyclotomicGaloisGroupOps;
use fheanor::number_ring::hypercube::isomorphism::HypercubeIsomorphism;
use fheanor::number_ring::hypercube::structure::HypercubeStructure;
use fheanor::number_ring::{AbstractNumberRing, NumberRingQuotientStore};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Layer;
use tracing_subscriber::util::SubscriberInitExt;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZi64: StaticRing<i64> = StaticRing::RING;

///
/// Returns `p^e/t`, where `p^e` is the characteristic of `P` and `t` is the CLPX modulus.
/// 
fn compute_scale<NumberRing, ZnTy, A, C>(P: &CLPXPlaintextRing<NumberRing, ZnTy, A, C>) -> El<ZnTy>
    where NumberRing: AbstractNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: NiceZn,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<ZnTy::Type>
{
    let ZZX = DensePolyRing::new(ZZbig, "X");
    let pe = int_cast(P.base_ring().integer_ring().clone_el(P.base_ring().modulus()), ZZbig, P.base_ring().integer_ring());
    let normt_over_pe = ZZbig.checked_div(P.get_ring().normt(), &pe).unwrap();
    let hom = P.base_ring().can_hom(&ZZbig).unwrap();
    let result = P.from_canonical_basis_extended((0..=ZZX.degree(P.get_ring().normt_t_inv()).unwrap())
        .map(|i| ZZbig.checked_div(ZZX.coefficient_at(P.get_ring().normt_t_inv(), i), &normt_over_pe).unwrap())
        .map(|x| hom.map(x)));
    assert!(P.wrt_canonical_basis(&result).iter().skip(1).all(|c| P.base_ring().is_zero(&c)));
    return P.wrt_canonical_basis(&result).at(0);
}

fn main() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    let filtered_chrome_layer = chrome_layer.with_filter(tracing_subscriber::filter::filter_fn(|metadata| !["small_basis_to_mult_basis", "mult_basis_to_small_basis", "small_basis_to_coeff_basis", "coeff_basis_to_small_basis"].contains(&metadata.name())));
    tracing_subscriber::registry().with(filtered_chrome_layer).init();

    let log2_m = 15;
    let params = Pow2CLPX::new(1 << log2_m);
    let number_ring = CLPXInstantiation::number_ring(&params);
    let acting_galois_group = number_ring.galois_group().get_group().clone().subgroup([number_ring.galois_group().from_representative(33)]);
    let ZZX = DensePolyRing::new(ZZbig, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(1 << (log2_m - 5)) - 2]);
    let t_sqr = ZZX.pow(ZZX.clone_el(&t), 2);
    let P2_clpx = CLPXInstantiation::create_plaintext_ring::<true>(&params, ZZX.clone(), ZZX.clone_el(&t_sqr), ZZbig.pow(int_cast(65537, ZZbig, ZZi64), 2), acting_galois_group.clone());
    let P1_clpx = CLPXInstantiation::create_plaintext_ring::<true>(&params, ZZX.clone(), ZZX.clone_el(&t), int_cast(65537, ZZbig, ZZi64), acting_galois_group.clone());
    let P1_bfv = BFVInstantiation::create_plaintext_ring(&params, ZZbig.clone_el(P1_clpx.base_ring().modulus()));
    let P2_bfv = BFVInstantiation::create_plaintext_ring(&params, ZZbig.clone_el(P2_clpx.base_ring().modulus()));
    let (C, C_mul) = CLPXInstantiation::create_ciphertext_rings(&params, 800..820, 1024);
    let digits = RNSGadgetVectorDigitIndices::select_digits(8, C.base_ring().len());

    // ======================== Prepare keys ======================== 
    
    let sk = <Pow2CLPX as CLPXInstantiation>::gen_sk(&C, rand::rng(), SecretKeyDistribution::SparseWithHwt(128));
    let C_reduced = RingValue::from(C.get_ring().drop_rns_factor(&RNSFactorIndexList::from(2..C.base_ring().len(), C.base_ring().len())));
    let sparse_sk = <Pow2CLPX as CLPXInstantiation>::gen_sk(&C_reduced, rand::rng(), SecretKeyDistribution::SparseWithHwt(32));
    let hom = P2_bfv.base_ring().can_hom(&ZZbig).unwrap();
    let enc_sparse_sk = <Pow2BFV as BFVInstantiation>::enc_sym(&P2_bfv, &C, rand::rng(), &P2_bfv.from_canonical_basis(C_reduced.wrt_canonical_basis(&sparse_sk).iter().map(|c| hom.map(C_reduced.base_ring().smallest_lift(c)))), &sk, 3.2);

    let h2_clpx = HypercubeStructure::default_pow2_hypercube(P2_clpx.acting_galois_group(), ZZbig.clone_el(P2_clpx.base_ring().modulus()));
    let H2_clpx = HypercubeIsomorphism::new::<true>(&&P2_clpx, &h2_clpx, Some("."));
    drop(h2_clpx);
    let H1_clpx = H2_clpx.change_modulus(&P1_clpx);
    let slots_to_coeffs = create_circuit_cached::<_, _, true>(&P1_clpx, &filename_keys!(slots_to_coeffs, m: P1_clpx.acting_galois_group().m(), o: P1_clpx.acting_galois_group().group_order(), p: ZZbig.clone_el(P1_clpx.base_ring().modulus())), Some("."), 
        || pow2::slots_to_coeffs_thin(&H1_clpx)
    );

    let poly_ring = DensePolyRing::new(P2_bfv.base_ring(), "X");
    let factor = H2_clpx.slot_ring().generating_poly(&poly_ring, P2_bfv.base_ring().can_hom(H2_clpx.slot_ring().base_ring()).unwrap());
    let h2_bfv = HypercubeStructure::default_pow2_hypercube(P2_bfv.acting_galois_group(), int_cast(*P2_bfv.base_ring().modulus(), ZZbig, ZZi64));
    let H2_bfv = HypercubeIsomorphism::new_with_poly_factor::<_, true>(&&P2_bfv, &poly_ring, &factor, &h2_bfv, Some("."));
    drop(h2_bfv);
    let coeffs_to_slots = create_circuit_cached::<_, _, true>(&P2_bfv, &filename_keys!(coeffs_to_slots, m: P2_bfv.acting_galois_group().m(), o: P2_bfv.acting_galois_group().group_order(), p: *P2_bfv.base_ring().modulus()), Some("."), 
        || pow2::coeffs_to_slots_thin(&H2_bfv)
    );

    let circuit = poly_to_circuit(&poly_ring, &[bounded_digit_retain_poly(&poly_ring, 6)]);

    let switch_to_sparse_key = <Pow2BFV as BFVInstantiation>::gen_switch_key(
        &C_reduced, 
        rand::rng(), 
        &<Pow2BFV as BFVInstantiation>::mod_switch_sk(&P1_bfv, &C_reduced, &C, &sk), 
        &sparse_sk, 
        &RNSGadgetVectorDigitIndices::select_digits(2, C_reduced.base_ring().len()), 
        3.2
    );

    // ======================== CLPX slots-to-coeffs ======================== 

    let m = H1_clpx.from_slot_values(
        [H1_clpx.slot_ring().int_hom().map(10000)].into_iter().chain((1..H1_clpx.slot_count()).map(|_| H1_clpx.slot_ring().zero()))
    );
    let ct_input = <Pow2CLPX as CLPXInstantiation>::enc_sym(&P1_clpx, &C, rand::rng(), &m, &sk, 3.2);
    
    println!("Input noise budget: {}", <Pow2CLPX as CLPXInstantiation>::noise_budget(&P1_clpx, &C, &ct_input, &sk));

    let gks = slots_to_coeffs.required_galois_keys(P1_clpx.acting_galois_group()).into_iter()
        .map(|g| {
            let gk = <Pow2CLPX as CLPXInstantiation>::gen_gk(&P1_clpx, &C, rand::rng(), &sk, &g, &digits, 3.2);
            (g, gk)
        })
        .collect::<Vec<_>>();
    let ct_in_slots = slots_to_coeffs.evaluate_clpx::<Pow2CLPX, _>(&P1_clpx, &P1_clpx, &C, None, &[ct_input], None, &gks, &mut 0, None).pop().unwrap();
    
    println!("After slots-to-coeffs: {}", <Pow2CLPX as CLPXInstantiation>::noise_budget(&P1_clpx, &C, &ct_in_slots, &sk));

    // ======================== Noisy expansion ========================

    let ct_sparse_key_switched = <Pow2BFV as BFVInstantiation>::key_switch(
        &C_reduced, 
        <Pow2BFV as BFVInstantiation>::mod_switch_ct(&P1_bfv, &C_reduced, &C, ct_in_slots), 
        &switch_to_sparse_key
    );
    let as_plain = <Pow2BFV as BFVInstantiation>::mod_switch_to_plaintext(&P2_bfv, &C_reduced, ct_sparse_key_switched);
    let ct_noisy_expanded = <Pow2BFV as BFVInstantiation>::hom_add_plain(&P2_bfv, &C, &as_plain.0, 
        <Pow2BFV as BFVInstantiation>::hom_mul_plain(&P2_bfv, &C, &as_plain.1, enc_sparse_sk)
    );

    println!("After noisy-expansion: {}", <Pow2BFV as BFVInstantiation>::noise_budget(&P2_bfv, &C, &ct_noisy_expanded, &sk));

    // ======================== BFV coeffs-to-slots ========================
    
    let gks = coeffs_to_slots.required_galois_keys(P2_bfv.acting_galois_group()).into_iter()
        .map(|g| {
            let gk = <Pow2BFV as BFVInstantiation>::gen_gk(&C, rand::rng(), &sk, &g, &digits, 3.2);
            (g, gk)
        })
        .collect::<Vec<_>>();
    let ct_bfv_slots = coeffs_to_slots.evaluate_bfv::<Pow2BFV, _>(&P2_bfv, &P2_bfv, &C, None, &[ct_noisy_expanded], None, &gks, &mut 0, None).pop().unwrap();

    println!("After coeffs-to-slots: {}", <Pow2BFV as BFVInstantiation>::noise_budget(&P2_bfv, &C, &ct_bfv_slots, &sk));

    // ======================== Back to CLPX ========================

    let bfv_per_clpx_slots = H2_bfv.slot_count() / H2_clpx.slot_count();
    
    // next, we compute the scale that is introduced when switching from BFV
    // to CLPX; Note that for `p^2/t^2 | a`, we have `q/p^2 a = q/t^2 (t^2 a/p^2 mod t^2)`;
    // hence we need to take care of the factor `t^2/p^2 mod t'`, where `t'` is
    // the effective modulus of `P2_clpx`
    let mask = H2_bfv.from_slot_values((0..H2_clpx.slot_count()).flat_map(|_| [H2_bfv.slot_ring().one()].into_iter().chain((1..bfv_per_clpx_slots).map(|_| H2_bfv.slot_ring().zero()))));
    let mut cts_bfv_slots_masked = (0..(bfv_per_clpx_slots / 2)).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| {
        let g = H2_bfv.hypercube().map(&[-(i as i64), -(j as i64)]);
        <Pow2BFV as BFVInstantiation>::hom_mul_plain(&P2_bfv, &C, &mask, 
            <Pow2BFV as BFVInstantiation>::hom_galois(&C, <Pow2BFV as BFVInstantiation>::clone_ct(&C, &ct_bfv_slots), &g, &<Pow2BFV as BFVInstantiation>::gen_gk(&C, rand::rng(), &sk, &g, &digits, 3.2))
        )
    }).collect::<Vec<_>>();

    // coeffs-to-slots put coefficients into slots in bitreversed order
    assert_eq!(1 << 4, cts_bfv_slots_masked.len());
    permute(&mut cts_bfv_slots_masked, |i| bitreverse(i, 4));

    let scale = P2_clpx.inclusion().map(compute_scale(&P2_clpx));
    let cts_clpx_slots = cts_bfv_slots_masked.into_iter().map(|ct| <Pow2CLPX as CLPXInstantiation>::hom_mul_plain(&P2_clpx, &C, &scale, ct)).collect::<Vec<_>>();
    
    println!("After convert-to-clpx: {}", <Pow2CLPX as CLPXInstantiation>::noise_budget(&P2_clpx, &C, &cts_clpx_slots[0], &sk));

    // ======================== Digit Extraction ========================

    let rk = <Pow2CLPX as CLPXInstantiation>::gen_rk(&C, rand::rng(), &sk, &digits, 3.2);
    
    // the ciphertexts `ct_cleaned` now contain, in their slots, 
    // the coefficients of `p^2/t m` modulo `p^2`
    let ct_cleaned = cts_clpx_slots.into_iter().map(|ct| {
        let ct_error = circuit.evaluate_clpx::<Pow2CLPX, _>(ZZi64, &P2_clpx, &C, Some(&C_mul), from_ref(&ct), Some(&rk), &[], &mut 0, None).pop().unwrap();
        return <Pow2CLPX as CLPXInstantiation>::hom_sub(&C, ct, &ct_error);
    }).collect::<Vec<_>>();

    println!("After digit-extract: {}", <Pow2CLPX as CLPXInstantiation>::noise_budget(&P2_clpx, &C, &ct_cleaned[0], &sk));

    // ======================== Modulo t ========================

    let hom = P1_clpx.inclusion().compose(P1_clpx.base_ring().can_hom(&ZZbig).unwrap());
    let ct_mod_t = ct_cleaned.into_iter().enumerate().map(|(i, ct)|
        <Pow2CLPX as CLPXInstantiation>::hom_mul_plain(&P1_clpx, &C, &hom.map(ZZbig.power_of_two(i)), ct)
    ).reduce(|ct1, ct2|
        <Pow2CLPX as CLPXInstantiation>::hom_add(&C, ct1, &ct2)
    ).unwrap();

    let scale = P1_clpx.inclusion().map(P1_clpx.base_ring().pow(P1_clpx.base_ring().invert(&compute_scale(&P1_clpx)).unwrap(), 2));
    let ct_result = <Pow2CLPX as CLPXInstantiation>::hom_mul_plain(&P1_clpx, &C, &scale, ct_mod_t);

    println!("After modulo-t: {}", <Pow2CLPX as CLPXInstantiation>::noise_budget(&P1_clpx, &C, &ct_result, &sk));

    let result = <Pow2CLPX as CLPXInstantiation>::dec(&P1_clpx, &C, ct_result, &sk);
    for x in H1_clpx.get_slot_values(&result) {
        H1_clpx.slot_ring().println(&x);
    }
}
