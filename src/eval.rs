use std::marker::PhantomData;
use std::mem::replace;

use feanor_math::algorithms::lll::float::lll;
use feanor_math::integer::{int_cast, BigIntRing};
use feanor_math::assert_el_eq;
use feanor_math::matrix::OwnedMatrix;
use feanor_math::ring::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::homomorphism::Homomorphism;
use feanor_math::group::AbelianGroupStore;
use feanor_math::seq::VectorView;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::seq::VectorFn;
use feanor_math::pid::EuclideanRingStore;

use fheanor::bfv::*;
use fheanor::bgv::SecretKeyDistribution;
use fheanor::circuit::evaluator::CircuitEvaluator;
use fheanor::circuit::{Coefficient, PlaintextCircuit};
use fheanor::number_ring::galois::GaloisGroupEl;
use fheanor::number_ring::NumberRingQuotientStore;

use crate::{ZZbig, ZZi64};

pub struct ScaledVectorBFVEvaluator<'b, Params: BFVInstantiation> {
    beta: i64,
    beta_wrapover: i64,
    len: usize,
    gks: &'b [(GaloisGroupEl, KeySwitchKey<Params>)],
    C: &'b CiphertextRing<Params>,
    P: &'b PlaintextRing<Params>,
    params: PhantomData<Params>
}

impl<'b, Params: BFVInstantiation> ScaledVectorBFVEvaluator<'b, Params> {

    pub fn new(beta: i64, len: usize, P: &'b PlaintextRing<Params>, C: &'b CiphertextRing<Params>, gks: &'b [(GaloisGroupEl, KeySwitchKey<Params>)]) -> Self {
        assert!(len > 0);
        let t = int_cast(P.base_ring().integer_ring().clone_el(P.base_ring().modulus()), ZZbig, P.base_ring().integer_ring());
        let beta_l = ZZbig.pow(int_cast(beta, ZZbig, ZZi64), len);
        // assert!(ZZbig.is_geq(&beta_l, &t));
        let beta_wrapover = int_cast(P.base_ring().smallest_lift(P.base_ring().coerce(&ZZbig, beta_l)), ZZi64, P.base_ring().integer_ring());
        assert!(beta_wrapover.abs() < 1000, "large beta^l % t will lead to poor noise growth");
        Self { P, C, len, beta, beta_wrapover, gks, params: PhantomData }
    }
}

impl<'a, 'b, Params: BFVInstantiation> CircuitEvaluator<'a, Vec<Ciphertext<Params>>, <Params as BFVInstantiation>::PlaintextRing> for ScaledVectorBFVEvaluator<'b, Params> {

    fn supports_gal(&self) -> bool { true }
    fn supports_mul(&self) -> bool { false }
    fn mul(&mut self, _: Vec<Ciphertext<Params>>, _: Vec<Ciphertext<Params>>) -> Vec<Ciphertext<Params>> { panic!("unsupported") }
    fn square(&mut self, _: Vec<Ciphertext<Params>>) -> Vec<Ciphertext<Params>> { panic!("unsupported") }

    fn constant(&mut self, constant: &'a Coefficient<<Params as BFVInstantiation>::PlaintextRing>) -> Vec<Ciphertext<Params>> {
        match constant {
            Coefficient::Zero => (0..self.len).map(|_| <Params as BFVInstantiation>::transparent_zero(self.C)).collect(),
            x => {
                let beta_mod = self.P.base_ring().coerce(&ZZi64, self.beta);
                let x = x.clone(self.P).to_ring_el(self.P);
                (0..self.len).map(|i| <Params as BFVInstantiation>::hom_add_plain(self.P, self.C, &self.P.inclusion().mul_ref_map(&x, &self.P.base_ring().pow(self.P.base_ring().clone_el(&beta_mod), i)), <Params as BFVInstantiation>::transparent_zero(self.C))).collect()
            }
        }
    }

    fn gal(&mut self, val: Vec<Ciphertext<Params>>, gs: &'a [GaloisGroupEl]) -> Vec<Vec<Ciphertext<Params>>> {
        let mut result = (0..gs.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for v in val {
            let gks = gs.as_fn().map_fn(|g1| &self.gks.iter().filter(|(g2, _)| self.C.acting_galois_group().eq_el(g1, g2)).next().expect("missing Galois key").1);
            let v_conjugates = if gs.len() > 1 {
                <Params as BFVInstantiation>::hom_galois_many(self.C, v, gs, gks)
            } else {
                vec![<Params as BFVInstantiation>::hom_galois(self.C, v, &gs[0], gks.at(0))]
            };
            for (v_conjugate, out) in v_conjugates.into_iter().zip(result.iter_mut()) {
                out.push(v_conjugate);
            }
        }
        return result;
    }

    fn add_inner_prod<'c, I>(&mut self, mut dst_parts: Vec<Ciphertext<Params>>, data: I) -> Vec<Ciphertext<Params>>
        where I: Iterator<Item = (&'a Coefficient<<Params as BFVInstantiation>::PlaintextRing>, &'c Vec<Ciphertext<Params>>)>,
            <Params as BFVInstantiation>::PlaintextRing: 'a,
            Vec<Ciphertext<Params>>: 'c
    {
        assert_eq!(self.len, dst_parts.len());
        let ZZ = self.P.base_ring().integer_ring();
        let beta = int_cast(self.beta, ZZ, ZZi64);
        let ZZ_to_Pbase = self.P.base_ring().can_hom(ZZ).unwrap();

        for (lhs, rhs_parts) in data {
            assert_eq!(self.len, rhs_parts.len());
            if let Coefficient::Zero = &lhs {
                continue;
            }

            let lhs = lhs.clone(self.P).to_ring_el(self.P);
            let mut lhs_lift = self.P.wrt_canonical_basis(&lhs).iter().map(|c| self.P.base_ring().smallest_lift(c)).collect::<Vec<_>>();
            let mut lhs_parts = Vec::new();
            for _ in 0..self.len {
                let part = self.P.from_canonical_basis(lhs_lift.iter_mut().map(|x| {
                    let (q, r) = ZZ.euclidean_div_rem(replace(x, ZZ.zero()), &beta);
                    *x = q;
                    return ZZ_to_Pbase.map(r);
                }));
                lhs_parts.push(part);
            }

            assert!(lhs_lift.iter().all(|c| ZZ.is_zero(c)));
            for i in 0..self.len {
                for j in 0..self.len {
                    if i + j < self.len {
                        dst_parts[i] = <Params as BFVInstantiation>::hom_add(self.C, 
                            <Params as BFVInstantiation>::hom_mul_plain(self.P, self.C, &lhs_parts[j], <Params as BFVInstantiation>::clone_ct(self.C, &rhs_parts[i + j])),
                            &dst_parts[i]
                        );
                    } else {
                        let factor = self.P.inclusion().mul_ref_map(&lhs_parts[j], &ZZ_to_Pbase.map(int_cast(self.beta_wrapover, self.P.base_ring().integer_ring(), ZZi64)));
                        dst_parts[i] = <Params as BFVInstantiation>::hom_add(self.C, 
                            <Params as BFVInstantiation>::hom_mul_plain(self.P, self.C, &factor, <Params as BFVInstantiation>::clone_ct(self.C, &rhs_parts[i + j - self.len])),
                            &dst_parts[i]
                        );
                    }
                }
            }
        }
        return dst_parts;
    }
}

#[test]
fn test_scaled_bfv_evaluator() {
    let params = Pow2BFV::new(512);
    let P = BFVInstantiation::create_plaintext_ring(&params, int_cast(65537, ZZbig, ZZi64));
    let (C, _) = BFVInstantiation::create_ciphertext_rings(&params, 400..420);

    let evaluator: ScaledVectorBFVEvaluator<Pow2BFV> = ScaledVectorBFVEvaluator::new(16, 4, &P, &C, &[]);
    let sk = <Pow2BFV as BFVInstantiation>::gen_sk(&C, rand::rng(), SecretKeyDistribution::UniformTernary);
    let cts = (0..4).map(|i| <Pow2BFV as BFVInstantiation>::enc_sym(&P, &C, rand::rng(), &P.inclusion().map(P.base_ring().coerce(&ZZi64, ZZi64.pow(16, i))), &sk, 3.2)).collect::<Vec<_>>();
    let circuit = PlaintextCircuit::linear_transform_ring(&[P.int_hom().map(257)], &P);
    let mut result = circuit.evaluate_generic(&[cts], evaluator).pop().unwrap().into_iter();
    assert_el_eq!(&P, P.int_hom().map(257), <Pow2BFV as BFVInstantiation>::dec(&P, &C, result.next().unwrap(), &sk));
    assert_el_eq!(&P, P.int_hom().map(257 * 16), <Pow2BFV as BFVInstantiation>::dec(&P, &C, result.next().unwrap(), &sk));
    assert_el_eq!(&P, P.int_hom().map(255), <Pow2BFV as BFVInstantiation>::dec(&P, &C, result.next().unwrap(), &sk));
    assert_el_eq!(&P, P.int_hom().map(255 * 16), <Pow2BFV as BFVInstantiation>::dec(&P, &C, result.next().unwrap(), &sk));
}

struct CvpSolver {
    lattice_lll: OwnedMatrix<El<BigIntRing>>
}

impl CvpSolver {

    fn new<R>(Zp: R, generators: &[El<R>])
        where R: RingStore,
            R::Type: ZnRing
    {
        lll();
    }
}
