# Changelog

## v0.16.0 (August 2024)

- **Bumped Version**: Updated version and README.
- **CI Updates**: Separated linting and testing actions.
- **Documentation**: Updated docs and improved handling of hidden features.
- **Bug Fixes**:
  - Fixed docstrings and embedding-net handling messages.
  - Resolved issues with batched sampling and associated tests (#35).
  - Fixed CI configurations and Pyright errors.
  - Corrected version retrieval process.
- **Enhancements**:
  - Added contribution guide (#5) and code of conduct.
  - Replaced setup with `pyproject.toml`.
  - Refactored CI; added pre-commit hooks and formatting.
  - Extended tests (#32).
- **License Change**: Switched to Apache 2.0 (#28).
- **Miscellaneous**: Performed Ruff checks and automatic fixes.

# v0.15.2
- Replace `torch.triangular_solve()` with `torch.linalg.solve_triangular` (#26).

# v0.15.1
- Fixes to version v0.15.0, in which the release did not contain the desired fixes.

# v0.15.0
- `epsilon` is added after multiplication of Cholesky-factors for better numerical stability (#22)

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
