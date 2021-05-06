# v0.14.2
- fix device bug in analytical MoG sampling (#19).

# v0.14.1
- refactor MoG to allow evaluation with analytically determined parameters (#17).

# v0.14.0
- Upgrade dependency on `nflows` to 0.14 for NSF fix (nflows #25).

# v0.13.1
- Fix GPU bug in MDN. 

# v0.13.0
- Upgrade dependency on `nflows` to 0.13 to allow GPU support. 

# v0.12.0
- Upgraded dependency on `nflows` to 0.12; explicitly allow Python 3.6 (#13)
- Support for `batch_x` with only one feature in mixture density networks (thanks to
  Harry Fu, #12)
- Imports from `nflows.transforms.splines` now available (thanks to Miles Cranmer, #11)