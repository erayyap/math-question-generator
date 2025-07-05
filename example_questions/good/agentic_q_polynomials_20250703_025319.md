# Generated Math Question

**Topic:** Polynomials

## Question

```
Let P(x) and R(x) be two distinct monic sixth–degree polynomials with integer coefficients that satisfy all the conditions below.

1. Both polynomials contain the same quadratic factor

  x² − m x + n  (with positive integers m,n ≤ 30)

   whose discriminant m² − 4n is not a perfect square (so the two
   roots of the quadratic are irrational).

2. After removing that common quadratic, the remaining factor of each
   polynomial has exactly four distinct integer roots, all lying in the
   interval [ −20, 20 ]; these four roots are the same for P and for R.

3. In addition to the four shared integer roots, P has one further
   integer root γ, and R has one further integer root δ, again in
   [ −20, 20 ], with γ ≠ δ.

4. The polynomials take the following values:

  P(0) = 144, P(3) = 0, P(4) = −480,

  R(0) = −288, R(2) = 0, R(5) ≡ 12 (mod 64).

Determine γ + δ.
```

## Solution

```
(Internal outline – not visible to students)

Write the common quadratic as Q(x)=x²−mx+n and the four shared integer
roots as α, β, σ, τ.  Then

 P(x)=Q(x)(x−α)(x−β)(x−σ)(x−τ)(x−γ),  
 R(x)=Q(x)(x−α)(x−β)(x−σ)(x−τ)(x−δ),

with all seven integer roots distinct and contained in [−20,20].

1. Constant terms  
 P(0)=Q(0)(−α)(−β)(−σ)(−τ)(−γ)=−n·(αβστ)·γ=144,  
 R(0)=−n·(αβστ)·δ=−288.

 Hence γ/δ=144/(−288)=−½ and δ=−2γ. (★)

2. Zero conditions at x=3 and x=2  
 Q(3)=3²−3m+n=9−3m+n ≠0 (since Q has irrational roots),  
 so P(3)=0 implies 3 is one of α,β,σ,τ,γ.  
 Analogously Q(2)=4−2m+n ≠0, hence R(2)=0 forces 2 to be among  
 α,β,σ,τ,δ.  Because γ and δ differ, 3 and 2 must both belong to the
 set {α,β,σ,τ}.  So the four shared integer roots already include 2
 and 3.

3. Divisibility filter from the constant terms  
 Put S=αβστ.  From P(0)=−nSγ we have  nS | 144.  Because n ≤ 30,
 only small values of S are possible; exhaustive checking of all
 4-tuples {α,β,σ,τ}⊂[−20,20] containing 2 and 3 shows that the only
 combination compatible with (★), the size bounds, and P(4)=−480 is

  {α,β,σ,τ} = {−4, −1, 2, 3},  S = 24.

4. Determining the quadratic factor  
 With S=24 the equalities P(0)=144 and R(0)=−288 give

  −n·24·γ = 144 ⇒ n·γ = −6,  
  −n·24·δ = −288 ⇒ n·δ = 12.

 Using δ = −2γ from (★) gives n = 3, γ = −2, δ = 4.
 The discriminant test m²−4·3 non-square and the value
 Q(4)·(4−γ)=−1·6 = −6 reproduce P(4)=−480, fixing m = 5.

5. Mod-64 check at x=5  
 R(5)=Q(5)(5−α)(5−β)(5−σ)(5−τ)(5−δ)
   =3·3·2·9·6·1=972≡12 (mod 64), matching the given congruence.

All conditions are now satisfied uniquely, so

 γ + δ = (−2) + 4 = 2.
```
