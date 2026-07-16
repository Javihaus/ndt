# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (latest) | Yes |
| < 0.1.0 | No |

Only the latest release of `ndtracker` receives security patches.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Report privately through either channel:

- **GitHub:** use *Security → Advisories → Report a vulnerability* on this repository
  (<https://github.com/Javihaus/ndt/security/advisories/new>).
- **Email:** **javier@jmarin.info**

Please include:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential impact assessment

### What to expect

- **Acknowledgment:** within 48 hours of your report.
- **Initial assessment:** within 5 business days. We will confirm whether the report is accepted, request additional information, or explain why it does not qualify.
- **Fix timeline:** critical vulnerabilities within 7 days, high severity within 14 days, moderate within 30 days.
- **Disclosure:** we coordinate disclosure timing with you, follow responsible-disclosure practices, and credit reporters unless they prefer anonymity.

## Scope

The following are in scope:

- **Code execution vulnerabilities** in the `ndt` core, trackers, or CLI/plotting utilities
- **Deserialization attacks** via crafted results files loaded by the library (CSV, JSON, HDF5)
- **Path traversal** through user-supplied export or results file paths
- **Dependency vulnerabilities** in direct dependencies (numpy, PyTorch, h5py, matplotlib, plotly) that affect `ndtracker` users

The following are out of scope:

- Vulnerabilities in PyTorch itself or other upstream frameworks -- report these to the respective projects
- Correctness of dimensionality metrics or jump detection (a research concern, not a security concern)
- Resource exhaustion from tracking very large models at high frequency (bounded by the user's own configuration)

## Security practices

- **No secrets handled.** `ndt` stores and transmits no credentials or API keys.
- **No network access for core functionality.** Tracking, metrics and export run entirely offline; network is only used by `pip` at install time.
- **Type safety.** Static type checking with mypy runs in CI.
- **Input validation.** Public functions validate their inputs before processing.
- **Pinned CI actions.** GitHub Actions are pinned to commit SHAs, and supply-chain posture is monitored with OpenSSF Scorecard.
