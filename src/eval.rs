use std::marker::PhantomData;
use std::mem::replace;

use feanor_math::algorithms::lll::float::lll;
use feanor_math::algorithms::matmul::{MatmulAlgorithm, STANDARD_MATMUL};
use feanor_math::field::*;
use feanor_math::integer::*;
use feanor_math::matrix::*;
use feanor_math::ring::*;
use feanor_math::rings::approx_real::float::Real64;
use feanor_math::algorithms::qr::QRDecompositionField;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::homomorphism::Homomorphism;
use feanor_math::group::AbelianGroupStore;
use feanor_math::seq::VectorView;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::seq::VectorFn;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::divisibility::*;

use fheanor::bfv::*;
use fheanor::circuit::evaluator::CircuitEvaluator;
use fheanor::circuit::*;
use fheanor::number_ring::galois::GaloisGroupEl;
use fheanor::number_ring::NumberRingQuotientStore;

use crate::{ZZbig, ZZi64};

pub struct GadgetDecomposingEvaluator<'b, Params: BFVInstantiation> {
    gadget_vector: GeneralGadgetVector<RingRef<'b, Params::PlaintextZnRing>>,
    gks: &'b [(GaloisGroupEl, KeySwitchKey<Params>)],
    C: &'b CiphertextRing<Params>,
    P: &'b PlaintextRing<Params>,
    params: PhantomData<Params>
}

impl<'b, Params: BFVInstantiation> GadgetDecomposingEvaluator<'b, Params> {

    pub fn new(gadget_vector: &[<Params::PlaintextZnRing as RingBase>::Element], P: &'b PlaintextRing<Params>, C: &'b CiphertextRing<Params>, gks: &'b [(GaloisGroupEl, KeySwitchKey<Params>)]) -> Self {
        let gadget_vector = GeneralGadgetVector::new(RingRef::new(P.base_ring().get_ring()), gadget_vector);
        let params = PhantomData;
        return Self { gadget_vector, P, C, gks, params };
    }
}

impl<'a, 'b, Params: BFVInstantiation> CircuitEvaluator<'a, Vec<Ciphertext<Params>>, <Params as BFVInstantiation>::PlaintextRing> for GadgetDecomposingEvaluator<'b, Params> {

    fn supports_gal(&self) -> bool { true }
    fn supports_mul(&self) -> bool { false }
    fn mul(&mut self, _: Vec<Ciphertext<Params>>, _: Vec<Ciphertext<Params>>) -> Vec<Ciphertext<Params>> { panic!("unsupported") }
    fn square(&mut self, _: Vec<Ciphertext<Params>>) -> Vec<Ciphertext<Params>> { panic!("unsupported") }

    fn constant(&mut self, constant: &'a Coefficient<<Params as BFVInstantiation>::PlaintextRing>) -> Vec<Ciphertext<Params>> {
        match constant {
            Coefficient::Zero => (0..self.gadget_vector.len()).map(|_| <Params as BFVInstantiation>::transparent_zero(self.C)).collect(),
            x => {
                let x = x.clone(self.P).to_ring_el(self.P);
                self.gadget_vector.as_iter().map(|c| <Params as BFVInstantiation>::hom_add_plain(self.P, self.C, &self.P.inclusion().mul_ref_map(&x, &c), <Params as BFVInstantiation>::transparent_zero(self.C))).collect()
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
        assert_eq!(self.gadget_vector.len(), dst_parts.len());
        let ZZ_to_Pbase = self.P.base_ring().can_hom(self.P.base_ring().integer_ring()).unwrap();

        for (lhs, rhs_parts) in data {
            assert_eq!(self.gadget_vector.len(), rhs_parts.len());
            if let Coefficient::Zero = &lhs {
                continue;
            }

            let lhs = lhs.clone(self.P).to_ring_el(self.P);
            let lhs_wrt_basis = self.P.wrt_canonical_basis(&lhs);
            let mut lhs_scaled_parts = (0..self.gadget_vector.len()).map(|_| (0..self.gadget_vector.len()).map(|_| Vec::new()).collect::<Vec<_>>()).collect::<Vec<_>>();
            for c in lhs_wrt_basis.iter() {
                for (i, gi) in self.gadget_vector.as_iter().enumerate() {
                    let decomposition = self.gadget_vector.gadget_decompose(&self.P.base_ring().mul_ref(&c, gi));
                    for (j, x) in decomposition.into_iter().enumerate() {
                        lhs_scaled_parts[i][j].push(x);
                    }
                }
            }
            let lhs_scaled_parts = lhs_scaled_parts.into_iter().map(|decomposition| decomposition.into_iter().map(|coeffs| 
                self.P.from_canonical_basis(coeffs.into_iter().map(|x| ZZ_to_Pbase.map(x)))
            ).collect::<Vec<_>>()).collect::<Vec<_>>();

            for (i, parts) in lhs_scaled_parts.into_iter().enumerate() {
                for (l, r) in parts.into_iter().zip(rhs_parts.iter()) {
                    dst_parts[i] = Params::hom_add(self.C, Params::hom_mul_plain(self.P, self.C, &l, Params::clone_ct(self.C, r)), &dst_parts[i]);
                }
            }
        }
        return dst_parts;
    }
}

///
/// Represents a general gadget vector, without any special structure to
/// compute gadget decompositions.
/// 
/// Don't use this with two many generators, as a CVP solver is required
/// to compute gadget decompositions when we don't have special structure.
/// Currently, the CVP solver is based on LLL, and thus can give quite bad
/// (i.e. large) results when the dimension of the lattice becomes large.
/// 
pub struct GeneralGadgetVector<R>
    where R: RingStore,
        R::Type: ZnRing
{
    generators: Vec<El<R>>,
    ring: R,
    lattice: OwnedMatrix<El<<R::Type as ZnRing>::IntegerRing>>,
    isometry: OwnedMatrix<f64>,
    triangular_basis: OwnedMatrix<f64>
}

impl<R> GeneralGadgetVector<R>
    where R: RingStore,
        R::Type: ZnRing
{
    pub fn new(ring: R, generators: &[El<R>]) -> Self {
        let ZZ = ring.integer_ring();
        assert!(ring.is_unit(&generators[0]));
        let n = generators.len();
        let mut lattice = OwnedMatrix::from_fn(n, n, |i, j| if i == 0 && j == n - 1 {
            ZZ.clone_el(ring.modulus())
        } else if i == 0 {
            ZZ.negate(ring.smallest_lift(ring.checked_div(&generators[j + 1], &generators[0]).unwrap()))
        } else if i == j + 1 {
            ZZ.one()
        } else {
            ZZ.zero()
        });
        lll(lattice.data_mut(), Real64::RING.can_hom(&ZZ).unwrap(), &0.99, &0.501, ()).unwrap();
        let mut triangular_basis = OwnedMatrix::from_fn(n, n, |i, j| ZZ.to_float_approx(lattice.at(i, j)));
        let mut isometry = OwnedMatrix::identity(n, n, Real64::RING);
        Real64::RING.get_ring().qr_decomposition(triangular_basis.data_mut(), isometry.data_mut());
        println!("GSO coefficients to use for CVP solving");
        println!("{}", format_matrix(n, n, |i, j| triangular_basis.at(i, j), Real64::RING));
        return Self {
            lattice: lattice,
            triangular_basis: triangular_basis,
            isometry: isometry,
            generators: generators.iter().map(|x| ring.clone_el(x)).collect(),
            ring: ring
        };
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn gadget_decompose(&self, target: &El<R>) -> Vec<El<<R::Type as ZnRing>::IntegerRing>> {
        let n = self.triangular_basis.row_count();
        let RR = Real64::RING;
        let ZZ = self.ring.integer_ring();
        let syndrome = self.ring.smallest_lift(self.ring().checked_div(target, &self.generators[0]).unwrap());
        let mut current = (0..n).map(|i| RR.mul(*self.isometry.at(0, i), ZZ.to_float_approx(&syndrome))).collect::<Vec<_>>();
        let mut result_idx = (0..n).map(|_| ZZ.zero()).collect::<Vec<_>>();
        for i in (0..n).rev() {
            let idx = RR.div(&current[i], self.triangular_basis.at(i, i)).round() as i64;
            result_idx[i] = int_cast(idx, ZZ, ZZi64);
            for j in 0..(i + 1) {
                RR.sub_assign(&mut current[j], RR.mul(idx as f64, *self.triangular_basis.at(j, i)));
            }
        }
        let mut result = (0..n).map(|_| ZZ.zero()).collect::<Vec<_>>();
        STANDARD_MATMUL.matmul(
            TransposableSubmatrix::from(self.lattice.data()),
            TransposableSubmatrix::from(Submatrix::from_1d(&result_idx, n, 1)),
            TransposableSubmatrixMut::from(SubmatrixMut::from_1d(&mut result, n, 1)),
            ZZ
        );
        for x in &mut result {
            ZZ.negate_inplace(x);
        }
        ZZ.add_assign(&mut result[0], syndrome);
        return result;
    }
}

impl<R> VectorView<El<R>> for GeneralGadgetVector<R>
    where R: RingStore,
        R::Type: ZnRing
{
    fn len(&self) -> usize {
        self.generators.len()
    }

    fn at(&self, i: usize) -> &El<R> {
        &self.generators[i]
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use feanor_math::algorithms::matmul::ComputeInnerProduct;

#[test]
fn test_scaled_bfv_evaluator() {
    let params = Pow2BFV::new(512);
    let P = BFVInstantiation::create_plaintext_ring(&params, int_cast(65537, ZZbig, ZZi64));
    let (C, _) = BFVInstantiation::create_ciphertext_rings(&params, 400..420);

    let evaluator: GadgetDecomposingEvaluator<Pow2BFV> = GadgetDecomposingEvaluator::new(&[1, 16 , 256, 4096].into_iter().map(|x| P.base_ring().int_hom().map(x)).collect::<Vec<_>>(), &P, &C, &[]);
    let sk = <Pow2BFV as BFVInstantiation>::gen_sk(&C, rand::rng(), SecretKeyDistribution::UniformTernary);
    let cts = (0..4).map(|i| <Pow2BFV as BFVInstantiation>::enc_sym(&P, &C, rand::rng(), &P.inclusion().map(P.base_ring().coerce(&ZZi64, ZZi64.pow(16, i))), &sk, 3.2)).collect::<Vec<_>>();
    let circuit = PlaintextCircuit::linear_transform_ring(&[P.int_hom().map(257)], &P);
    let mut result = circuit.evaluate_generic(&[cts], evaluator).pop().unwrap().into_iter();
    assert_el_eq!(&P, P.int_hom().map(257), <Pow2BFV as BFVInstantiation>::dec(&P, &C, result.next().unwrap(), &sk));
    assert_el_eq!(&P, P.int_hom().map(257 * 16), <Pow2BFV as BFVInstantiation>::dec(&P, &C, result.next().unwrap(), &sk));
    assert_el_eq!(&P, P.int_hom().map(255), <Pow2BFV as BFVInstantiation>::dec(&P, &C, result.next().unwrap(), &sk));
    assert_el_eq!(&P, P.int_hom().map(255 * 16), <Pow2BFV as BFVInstantiation>::dec(&P, &C, result.next().unwrap(), &sk));
}

#[test]
fn test_cvp_solver() {
    let Fp = Zn::new(65537);
    let ZZ_to_Fp = Fp.can_hom(&ZZi64).unwrap();
    let mixed_inner_prod = |a: &[El<Zn>], b: &[i64]| Fp.get_ring().inner_product(a.iter().copied().zip(b.iter().map(|x| ZZ_to_Fp.map(*x))));

    let generators = [ZZ_to_Fp.map(1), ZZ_to_Fp.map(256)];
    let solver = GeneralGadgetVector::new(Fp, &generators);
    let combination = solver.gadget_decompose(&ZZ_to_Fp.map(500));
    assert!(combination[0].abs() <= 12);
    assert!(combination[1].abs() <= 12);
    assert_el_eq!(&Fp, ZZ_to_Fp.map(500), mixed_inner_prod(&generators, &combination));

    let generators = [1, 16, 256, 4096, 65536, 65521, 65281, 61441].into_iter().map(|x| ZZ_to_Fp.map(x)).collect::<Vec<_>>();
    let solver = GeneralGadgetVector::new(Fp, &generators);
    let combination = solver.gadget_decompose(&ZZ_to_Fp.map(500));
    assert!(combination.iter().all(|x| x.abs() <= 3));
    assert_el_eq!(&Fp, ZZ_to_Fp.map(500), mixed_inner_prod(&generators, &combination));
}