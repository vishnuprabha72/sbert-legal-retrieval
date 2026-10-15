"""
Preflight check for watsonx credentials.

A missing `.env`, or one still containing the placeholder values from
.env.example, otherwise only shows up as an IBM IAM error deep inside Stage 2,
after Stages 0 and 1 have already run.

Call check_credentials() at the very start of anything that will eventually
need watsonx, so a bad setup fails in under a second with an actionable
message instead of partway through a multi-judgment batch.
"""
import os
import sys

from dotenv import load_dotenv

_PLACEHOLDER_VALUES = {"your-key", "your-project-id", ""}
_REQUIRED = ["WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL"]


def check_credentials(exit_on_failure: bool = True) -> bool:
    load_dotenv()
    problems = []

    for var in _REQUIRED:
        value = os.environ.get(var)
        if not value:
            problems.append(f"  {var} is not set at all.")
        elif value.strip().strip('"').strip("'") in _PLACEHOLDER_VALUES:
            problems.append(f"  {var} is still the placeholder value ({value!r}) -- put your real value in .env.")

    if problems:
        message = (
            "\n--- watsonx credentials check FAILED ---\n"
            + "\n".join(problems) +
            "\n\nFix: edit .env in the project root with your real WATSONX_API_KEY, "
            "WATSONX_PROJECT_ID, WATSONX_URL.\n"
            "If you've fixed .env but this still fails, your SHELL likely has old "
            "values exported that are shadowing .env -- run:\n"
            "  env | grep WATSONX\n"
            "and if that shows anything, run:\n"
            "  unset WATSONX_API_KEY WATSONX_PROJECT_ID WATSONX_URL\n"
            "then try again in the SAME terminal (a new terminal only helps if the "
            "export was removed from a shell startup file like ~/.bashrc).\n"
            "-----------------------------------------\n"
        )
        print(message, file=sys.stderr)
        if exit_on_failure:
            sys.exit(1)
        return False

    return True


if __name__ == "__main__":
    # Standalone use: `uv run python src/common/check_env.py`
    ok = check_credentials(exit_on_failure=False)
    if ok:
        print("Credentials look OK (present, non-placeholder).")
        sys.exit(0)
    sys.exit(1)
