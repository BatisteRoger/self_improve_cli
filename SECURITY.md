# Security Policy

## Threat model

This project downloads, processes, and stores AI-agent execution traces. Traces can contain:

- Prompts and completions (which may include user content, instructions, or internal logic)
- Tool arguments and results (which may include file contents, database records, or API responses)
- Error messages (which may include stack traces, URLs, or credentials)
- Metadata (which may include project names, user identifiers, or internal paths)

### What we protect against

- **Accidental exposure of PII and secrets in local artifacts.** The CLI anonymizes traces on ingestion by default, replacing emails, phone numbers, financial identifiers, API keys, tokens, private keys, and connection strings with stable placeholders.
- **Accidental commits of sensitive data.** The `.gitignore` excludes `.env`, `data/`, and all local artifacts.

### What we do not protect against

- **All sensitive values.** No detector catches every secret or contextual identifier. Anonymization is defense in depth, not a guarantee.
- **Cross-trace linkability.** Placeholders are stable within one anonymization pass (one trace), not across traces. A global identity map is never persisted.
- **Side channels.** Logs, error messages, and debug output may still contain sensitive values that were not detected. Review logs before sharing.
- **Malicious actors with local access.** If an attacker has access to the local filesystem, they can access both sanitized and raw (opt-in) artifacts.

## Known limitations

These are harder to fix and are tracked as known vulnerabilities. They are acceptable for the current use case (local developer tool) but should be addressed before any broader deployment.

### 1. Anonymization coverage is incomplete

**Risk:** Sensitive values that don't match known patterns will pass through unredacted. This includes internal project names, custom identifiers, employee IDs, customer references, non-English PII, and novel secret formats.

**Root cause:** The regex fallback covers common patterns (email, phone, IBAN, credit card, SSN, API keys, bearer tokens, private keys, JWTs, connection strings, password assignments). Presidio adds NLP-based detection when installed but is not available by default and doesn't cover every entity type.

**Mitigation:** Always review sanitized output before sharing. The sanitization report includes `complete: false` if Presidio failed mid-analysis. Custom recognizers can be added to the regex pattern list.

**Not fixed because:** Perfect anonymization is an open problem. The current coverage is a pragmatic 80% solution. A pluggable recognizer system is a future enhancement.

### 2. `run_detail` shows full untruncated content by design

**Risk:** The L3 `run-detail` command outputs full message text, tool arguments, and tool results without truncation. If the loaded trace was not properly sanitized (e.g., loaded from `raw.json`), this exposes unanonymized data.

**Current protection:** `load_trace` now raises `ValueError` if it loads an unsanitized raw trace. The `fetch` command anonymizes before persistence by default.

**Not fixed because:** L3 is intentionally the detailed inspection level. Truncating it would defeat its purpose. The privacy boundary is enforced upstream (at fetch and load time), not at display time.

### 3. No rate limiting or retry logic on API calls

**Risk:** A bug or script error could cause rapid repeated calls to the LangSmith API, potentially triggering rate limits or causing excessive data download.

**Not fixed because:** The CLI is a developer tool used interactively, not a production service. The LangSmith SDK has its own retry logic. Adding rate limiting here would add complexity without clear benefit for the current use case.

### 4. Presidio model download and platform constraints

**Risk:** Presidio depends on spaCy NLP models that must be downloaded separately. On platforms where spaCy wheels are unavailable (e.g., Python 3.14 at time of writing), Presidio cannot be installed and the CLI falls back to regex-only detection, which is less comprehensive.

**Mitigation:** The sanitization report's `recognizer_version` field indicates whether Presidio or regex-fallback was used. Users should be aware that regex-fallback provides weaker coverage.

**Not fixed because:** This is a dependency constraint, not a code issue. The fallback ensures the CLI remains functional without Presidio.

## Reporting a vulnerability

If you discover a security issue:

1. Do not open a public issue.
2. Contact the maintainers privately with details and a reproduction if possible.
3. Allow reasonable time for a fix before public disclosure.

## Safe usage guidelines

- Use a dedicated, least-privileged observability token.
- Use a non-sensitive project for testing and development.
- **Configure `LANGSMITH_ALLOWED_PROJECTS`** to restrict which projects the CLI can query. This prevents accidentally fetching from production. Leave empty to allow all (gated by API key), or set glob patterns like `staging*,my-agent-dev`.
- Never commit `.env`, `data/`, or any file under `data/`.
- Review all fixtures, examples, and screenshots for sensitive content before publishing.
- Use `--keep-raw` only when you understand the risks and have secured the local environment.
- Run secret scanning on the repository before publishing or accepting external contributions.
