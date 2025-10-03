#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![feature(allocator_api)]
#![feature(iter_array_chunks)]

use std::alloc::Allocator;
use std::array::from_ref;
use std::fmt::Display;
use std::marker::PhantomData;
use std::mem::replace;
use std::ops::Range;
use std::sync::Mutex;

use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::algorithms::fft::cooley_tuckey::bitreverse;
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::divisibility::*;
use feanor_math::group::AbelianGroupStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, BigIntRing, IntegerRingStore};
use feanor_math::matrix::{format_matrix, OwnedMatrix};
use feanor_math::ordered::OrderedRingStore;
use feanor_math::primitive_int::*;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::seq::permute::permute;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;

use feanor_math::rings::extension::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::*;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::rings::zn::*;
use feanor_math::seq::{VectorFn, VectorView};
use fheanor::bfv::{force_double_rns_repr, BFVInstantiation, Ciphertext, CiphertextRing, KeySwitchKey, PlaintextRing, Pow2BFV, SecretKey};
use fheanor::bgv::SecretKeyDistribution;
use fheanor::ciphertext_ring::double_rns_managed::ManagedDoubleRNSRingBase;
use fheanor::ciphertext_ring::indices::RNSFactorIndexList;
use fheanor::ciphertext_ring::{BGFVCiphertextRing, RNSFactorCongruence};
use fheanor::circuit::evaluator::CircuitEvaluator;
use fheanor::ntt::FheanorNegacyclicNTT;
use fheanor::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use fheanor::number_ring::quotient_by_int::NumberRingQuotientByIntBase;
use fheanor::rns_conv::matrix_lift::AlmostExactMatrixBaseConversion;
use fheanor::rns_conv::RNSOperation;
use fheanor::{circuit::*, DefaultCiphertextAllocator, DefaultNegacyclicNTT};
use fheanor::clpx::encoding::CLPXPlaintextRing;
use fheanor::clpx::{CLPXInstantiation, Pow2CLPX};
use fheanor::digit_extract::polys::{bounded_digit_retain_poly, poly_to_circuit};
use fheanor::filename_keys;
use fheanor::NiceZn;
use fheanor::gadget_product::digits::RNSGadgetVectorDigitIndices;
use fheanor::lin_transform::pow2;
use fheanor::number_ring::galois::{CyclotomicGaloisGroupOps, GaloisGroupEl};
use fheanor::number_ring::hypercube::isomorphism::HypercubeIsomorphism;
use fheanor::number_ring::hypercube::structure::HypercubeStructure;
use fheanor::number_ring::{extend_sampled_primes, largest_prime_leq_congruent_to_one, sample_primes, AbstractNumberRing, NumberRingQuotientStore};
use tracing::instrument;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Layer;
use tracing_subscriber::util::SubscriberInitExt;

use crate::eval::GadgetDecomposingEvaluator;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZi64: StaticRing<i64> = StaticRing::RING;
const ZZi128: StaticRing<i128> = StaticRing::RING;

mod eval;

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

#[derive(Debug)]
pub struct BigPow2BFV<A: Allocator + Clone  = DefaultCiphertextAllocator, N: FheanorNegacyclicNTT<Zn> = DefaultNegacyclicNTT> {
    number_ring: Pow2CyclotomicNumberRing<N>,
    ciphertext_allocator: A
}

impl BigPow2BFV {

    pub fn new(m: usize) -> Self {
        Self::new_with_ntt(m, DefaultCiphertextAllocator::default())
    }
}

impl<A: Allocator + Clone , N: FheanorNegacyclicNTT<Zn>> BigPow2BFV<A, N> {

    #[instrument(skip_all)]
    pub fn new_with_ntt(m: usize, allocator: A) -> Self {
        return Self {
            number_ring: Pow2CyclotomicNumberRing::new_with_ntt(m as u64),
            ciphertext_allocator: allocator
        }
    }
    
    pub fn ciphertext_allocator(&self) -> &A {
        &self.ciphertext_allocator
    }
}

impl<A: Allocator + Clone , C: FheanorNegacyclicNTT<Zn>> Display for BigPow2BFV<A, C> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BFV({:?})", self.number_ring)
    }
}

impl<A: Allocator + Clone , C: FheanorNegacyclicNTT<Zn>> Clone for BigPow2BFV<A, C> {

    fn clone(&self) -> Self {
        Self {
            number_ring: self.number_ring.clone(),
            ciphertext_allocator: self.ciphertext_allocator.clone()
        }
    }
}

impl<A: Allocator + Clone , C: FheanorNegacyclicNTT<Zn>> BFVInstantiation for BigPow2BFV<A, C> {

    type NumberRing = Pow2CyclotomicNumberRing<C>;
    type CiphertextRing = ManagedDoubleRNSRingBase<Pow2CyclotomicNumberRing<C>, A>;
    type PlaintextRing = NumberRingQuotientByIntBase<Pow2CyclotomicNumberRing<C>, zn_big::Zn<BigIntRing>>;
    type PlaintextZnRing = zn_big::ZnBase<BigIntRing>;

    #[instrument(skip_all)]
    fn number_ring(&self) -> &Pow2CyclotomicNumberRing<C> {
        &self.number_ring
    }

    #[instrument(skip_all)]
    fn create_plaintext_ring(&self, t: El<BigIntRing>) -> PlaintextRing<Self> {
        NumberRingQuotientByIntBase::new(self.number_ring().clone(), zn_big::Zn::new(ZZbig, t))
    }

    #[instrument(skip_all)]
    fn create_ciphertext_rings(&self, log2_q: Range<usize>) -> (CiphertextRing<Self>, CiphertextRing<Self>) {
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;
        let next_prime = |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZi64, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZi64));
        let C_rns_base_primes = sample_primes(log2_q.start, log2_q.end, 57, &next_prime).unwrap();
        let C_rns_base = zn_rns::Zn::new(C_rns_base_primes.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZi64, ZZbig) as u64)).collect::<Vec<_>>(), ZZbig);

        let Cmul_modulus_size = 2 * ZZbig.abs_log2_ceil(C_rns_base.modulus()).unwrap() + number_ring.product_expansion_factor().log2().ceil() as usize;
        let Cmul_rns_base_primes = extend_sampled_primes(&C_rns_base_primes, Cmul_modulus_size + 3, Cmul_modulus_size + 60, 57, &next_prime).unwrap();
        let Cmul_rns_base = zn_rns::Zn::new(Cmul_rns_base_primes.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZi64, ZZbig) as u64)).collect(), ZZbig);

        let C_mul = ManagedDoubleRNSRingBase::new_with_alloc(
            number_ring.clone(),
            Cmul_rns_base,
            self.ciphertext_allocator.clone()
        );

        let dropped_indices = RNSFactorIndexList::from((0..C_mul.base_ring().len()).filter(|i| C_rns_base.as_iter().all(|Zp| Zp.get_ring() != C_mul.base_ring().at(*i).get_ring())), C_mul.base_ring().len());
        let C = RingValue::from(C_mul.get_ring().drop_rns_factor(&dropped_indices));
        assert!(C.base_ring().get_ring() == C_rns_base.get_ring());
        return (C, C_mul);
    }

    #[instrument(skip_all)]
    fn encode_plain_multiplicant(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>) -> El<CiphertextRing<Self>> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        let result = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
        return force_double_rns_repr::<Self, _, _>(C, result);
    }
    
    fn lift_to_Cmul<'a>(C: &'a CiphertextRing<Self>, C_mul: &'a CiphertextRing<Self>) -> Box<dyn 'a + for<'b> FnMut(&'b El<CiphertextRing<Self>>) -> El<CiphertextRing<Self>>> {
        let C_delta = RingValue::from(C_mul.get_ring().drop_rns_factor(&RNSFactorIndexList::from(0..C.base_ring().len(), C_mul.base_ring().len())));
        let lift = AlmostExactMatrixBaseConversion::new(
            C.base_ring().as_iter().cloned().collect::<Vec<_>>(),
            C_delta.base_ring().as_iter().cloned().collect::<Vec<_>>()
        );
        let mut tmp_in = OwnedMatrix::zero(C.base_ring().len(), C_mul.get_ring().small_generating_set_len(), C_mul.base_ring().at(0));
        let mut tmp_out = OwnedMatrix::zero(C_mul.base_ring().len() - C.base_ring().len(), C_mul.get_ring().small_generating_set_len(), C_mul.base_ring().at(0));
        return Box::new(move |c| {
            C.get_ring().as_representation_wrt_small_generating_set(c, tmp_in.data_mut());
            lift.apply(tmp_in.data(), tmp_out.data_mut());
            let delta = force_double_rns_repr::<Self, _, _>(&C_delta, C_delta.get_ring().from_representation_wrt_small_generating_set(tmp_out.data()));
            return C_mul.get_ring().collect_rns_factors(
                (0..C.base_ring().len()).map(|i| RNSFactorCongruence::CongruentTo(C.get_ring(), i, c)).chain(
                    (0..C_delta.base_ring().len()).map(|i| RNSFactorCongruence::CongruentTo(C_delta.get_ring(), i, &delta))
                )
            );
        });
    }
}

fn main() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    let filtered_chrome_layer = chrome_layer.with_filter(tracing_subscriber::filter::filter_fn(|metadata| !["small_basis_to_mult_basis", "mult_basis_to_small_basis", "small_basis_to_coeff_basis", "coeff_basis_to_small_basis"].contains(&metadata.name())));
    tracing_subscriber::registry().with(filtered_chrome_layer).init();

    let log2_m = 10;
    let params = Pow2CLPX::new(1 << log2_m);
    let bfv_params = BigPow2BFV::new(1 << log2_m);
    let number_ring = CLPXInstantiation::number_ring(&params);
    let acting_galois_group = number_ring.galois_group().get_group().clone().subgroup([number_ring.galois_group().from_representative(129)]);
    let ZZX = DensePolyRing::new(ZZbig, "X");
    let [t] = ZZX.with_wrapped_indeterminate(|X| [X.pow_ref(4) - 2 * X + 2]);
    let t = ZZX.evaluate(&t, &ZZX.from_terms([(ZZbig.one(), 1 << (log2_m - 7))]), ZZX.inclusion());
    let t_sqr = ZZX.pow(ZZX.clone_el(&t), 2);
    let p = int_cast(381380816531458621441, ZZbig, ZZi128);
    let P1_clpx = CLPXInstantiation::create_plaintext_ring::<true>(&params, ZZX.clone(), ZZX.clone_el(&t), ZZbig.clone_el(&p), acting_galois_group.clone());
    let P2_clpx = CLPXInstantiation::create_plaintext_ring::<true>(&params, ZZX.clone(), ZZX.clone_el(&t_sqr), ZZbig.pow(ZZbig.clone_el(&p), 2), acting_galois_group.clone());
    let P1_bfv = BFVInstantiation::create_plaintext_ring(&bfv_params, ZZbig.clone_el(P1_clpx.base_ring().modulus()));
    let P2_bfv = BFVInstantiation::create_plaintext_ring(&bfv_params, ZZbig.clone_el(P2_clpx.base_ring().modulus()));
    let (C, C_mul) = CLPXInstantiation::create_ciphertext_rings(&params, 800..820, 1024);
    let digits = RNSGadgetVectorDigitIndices::select_digits(8, C.base_ring().len());

    // ======================== Prepare keys ======================== 

    let gadget_vector = (0..).scan(ZZbig.one(), |current, _| {
        let result = ZZbig.clone_el(current);
        ZZbig.int_hom().mul_assign_map(current, 1000);
        return Some(result);
    })
        .take_while(|x| ZZbig.is_lt(x, P2_bfv.base_ring().modulus()))
        .map(|x| P2_bfv.base_ring().coerce(&ZZbig, x))
        .collect::<Vec<_>>();
    
    let sk = <Pow2CLPX as CLPXInstantiation>::gen_sk(&C, rand::rng(), SecretKeyDistribution::SparseWithHwt(128));
    let C_reduced = RingValue::from(C.get_ring().drop_rns_factor(&RNSFactorIndexList::from(2..C.base_ring().len(), C.base_ring().len())));
    let sparse_sk = <Pow2CLPX as CLPXInstantiation>::gen_sk(&C_reduced, rand::rng(), SecretKeyDistribution::SparseWithHwt(32));
    let hom = P2_bfv.base_ring().can_hom(&ZZbig).unwrap();
    let enc_sparse_sk = gadget_vector.iter().map(|gi| 
        <BigPow2BFV as BFVInstantiation>::enc_sym(&P2_bfv, &C, rand::rng(), &P2_bfv.inclusion().mul_ref_map(
            &P2_bfv.from_canonical_basis(C_reduced.wrt_canonical_basis(&sparse_sk).iter().map(|c| hom.map(C_reduced.base_ring().smallest_lift(c)))),
            gi)
        , &sk, 3.2)
    ).collect::<Vec<_>>();

    let h2_clpx = HypercubeStructure::default_pow2_hypercube(P2_clpx.acting_galois_group(), ZZbig.clone_el(P2_clpx.base_ring().modulus()));
    let H2_clpx = HypercubeIsomorphism::new::<true>(&&P2_clpx, &h2_clpx, Some("."));
    drop(h2_clpx);
    let H1_clpx = H2_clpx.change_modulus(&P1_clpx);
    let slots_to_coeffs = create_circuit_cached::<_, _, true>(&P1_clpx, &filename_keys!{slots_to_coeffs, m: P1_clpx.acting_galois_group().m(), o: P1_clpx.acting_galois_group().group_order(), p: ZZbig.clone_el(P1_clpx.base_ring().modulus())}, Some("."), 
        || pow2::slots_to_coeffs_thin(&H1_clpx)
    );

    let poly_ring = DensePolyRing::new(P2_bfv.base_ring(), "X");
    let factor = H2_clpx.slot_ring().generating_poly(&poly_ring, P2_bfv.base_ring().can_hom(H2_clpx.slot_ring().base_ring()).unwrap());
    let h2_bfv = HypercubeStructure::default_pow2_hypercube(P2_bfv.acting_galois_group(), ZZbig.clone_el(P2_bfv.base_ring().modulus()));
    let H2_bfv = HypercubeIsomorphism::new_with_poly_factor::<_, true>(&&P2_bfv, &poly_ring, &factor, &h2_bfv, Some("."));
    drop(h2_bfv);
    let coeffs_to_slots = create_circuit_cached::<_, _, true>(&P2_bfv, &filename_keys!{coeffs_to_slots, m: P2_bfv.acting_galois_group().m(), o: P2_bfv.acting_galois_group().group_order(), p: ZZbig.clone_el(P2_bfv.base_ring().modulus())}, Some("."), 
        || pow2::coeffs_to_slots_thin(&H2_bfv)
    );

    let circuit = poly_to_circuit(&poly_ring, &[bounded_digit_retain_poly(&poly_ring, 6)]);

    let switch_to_sparse_key = <BigPow2BFV as BFVInstantiation>::gen_switch_key(
        &C_reduced, 
        rand::rng(), 
        &<BigPow2BFV as BFVInstantiation>::mod_switch_sk(&P1_bfv, &C_reduced, &C, &sk), 
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

    let ct_sparse_key_switched = <BigPow2BFV as BFVInstantiation>::key_switch(
        &C_reduced, 
        <BigPow2BFV as BFVInstantiation>::mod_switch_ct(&P1_bfv, &C_reduced, &C, ct_in_slots), 
        &switch_to_sparse_key
    );
    let as_plain = <BigPow2BFV as BFVInstantiation>::mod_switch_to_plaintext(&P2_bfv, &C_reduced, ct_sparse_key_switched);
    let ct_noisy_expanded = gadget_vector.iter().zip(enc_sparse_sk.iter()).map(|(gi, gi_sk)| 
            <BigPow2BFV as BFVInstantiation>::hom_add_plain(&P2_bfv, &C, &P2_bfv.inclusion().mul_ref_map(&as_plain.0, gi), 
            <BigPow2BFV as BFVInstantiation>::hom_mul_plain(&P2_bfv, &C, &as_plain.1, <BigPow2BFV as BFVInstantiation>::clone_ct(&C, &gi_sk))
        )
    ).collect::<Vec<_>>();

    println!("After noisy-expansion: {}", <BigPow2BFV as BFVInstantiation>::noise_budget(&P2_bfv, &C, &ct_noisy_expanded[0], &sk));

    // ======================== BFV coeffs-to-slots ========================
    
    let gks = coeffs_to_slots.required_galois_keys(P2_bfv.acting_galois_group()).into_iter()
        .map(|g| {
            let gk = <BigPow2BFV as BFVInstantiation>::gen_gk(&C, rand::rng(), &sk, &g, &digits, 3.2);
            (g, gk)
        })
        .collect::<Vec<_>>();
    let gadget_decomposing_evaluator = GadgetDecomposingEvaluator::<BigPow2BFV>::new(&gadget_vector, &P2_bfv, &C, &gks);
    let ct_bfv_slots = coeffs_to_slots.evaluate_generic(&[ct_noisy_expanded], gadget_decomposing_evaluator).pop().unwrap().into_iter().next().unwrap();

    println!("After coeffs-to-slots: {}", <BigPow2BFV as BFVInstantiation>::noise_budget(&P2_bfv, &C, &ct_bfv_slots, &sk));

    // ======================== Back to CLPX ========================

    let bfv_per_clpx_slots = H2_bfv.slot_count() / H2_clpx.slot_count();
    
    // next, we compute the scale that is introduced when switching from BFV
    // to CLPX; Note that for `p^2/t^2 | a`, we have `q/p^2 a = q/t^2 (t^2 a/p^2 mod t^2)`;
    // hence we need to take care of the factor `t^2/p^2 mod t'`, where `t'` is
    // the effective modulus of `P2_clpx`
    let mask = H2_bfv.from_slot_values((0..H2_clpx.slot_count()).flat_map(|_| [H2_bfv.slot_ring().one()].into_iter().chain((1..bfv_per_clpx_slots).map(|_| H2_bfv.slot_ring().zero()))));
    let mut cts_bfv_slots_masked = (0..(bfv_per_clpx_slots / 2)).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| {
        let g = H2_bfv.hypercube().map(&[-(i as i64), -(j as i64)]);
        <BigPow2BFV as BFVInstantiation>::hom_mul_plain(&P2_bfv, &C, &mask, 
            <BigPow2BFV as BFVInstantiation>::hom_galois(&C, <BigPow2BFV as BFVInstantiation>::clone_ct(&C, &ct_bfv_slots), &g, &<BigPow2BFV as BFVInstantiation>::gen_gk(&C, rand::rng(), &sk, &g, &digits, 3.2))
        )
    }).collect::<Vec<_>>();

    // coeffs-to-slots put coefficients into slots in bitreversed order
    let log2_bfv_per_clpx_slots = ZZi64.abs_log2_ceil(&(bfv_per_clpx_slots as i64)).unwrap();
    assert_eq!(1 << log2_bfv_per_clpx_slots, cts_bfv_slots_masked.len());
    permute(&mut cts_bfv_slots_masked, |i| bitreverse(i, log2_bfv_per_clpx_slots));

    let scale = P2_clpx.inclusion().map(compute_scale(&P2_clpx));
    let cts_clpx_slots = cts_bfv_slots_masked.into_iter().map(|ct| <Pow2CLPX as CLPXInstantiation>::hom_mul_plain(&P2_clpx, &C, &scale, ct)).collect::<Vec<_>>();
    
    println!("After convert-to-clpx: {}", <Pow2CLPX as CLPXInstantiation>::noise_budget(&P2_clpx, &C, &cts_clpx_slots[0], &sk));

    // ======================== Digit Extraction ========================

    let rk = <Pow2CLPX as CLPXInstantiation>::gen_rk(&C, rand::rng(), &sk, &digits, 3.2);
    
    // the ciphertexts `ct_cleaned` now contain, in their slots, 
    // the coefficients of `p^2/t m` modulo `p^2`
    let ct_cleaned = cts_clpx_slots.into_iter().map(|ct| {
        let ct_error = circuit.evaluate_clpx::<Pow2CLPX, _>(ZZbig, &P2_clpx, &C, Some(&C_mul), from_ref(&ct), Some(&rk), &[], &mut 0, None).pop().unwrap();
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
