"""Creates a user directly in the database (no running server needed).

    python -m app.create_user EMAIL [--name NAME] [--password PASSWORD]
    docker compose exec backend python -m app.create_user EMAIL

If --password is omitted it is asked for interactively (not echoed), which
keeps it out of shell history. Uses the same settings as the app, so
ROCKSEG_STORAGE_DIR / ROCKSEG_DATABASE_URL pick the target database.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from pydantic import ValidationError

from .core.security import hash_password
from .db import models
from .db.session import get_session_factory, init_db
from .schemas import RegisterRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("email")
    parser.add_argument("--name", help="display name (default: the part of the email before @)")
    parser.add_argument("--password", help="omit to be prompted")
    args = parser.parse_args(argv)

    password = args.password
    if password is None:
        password = getpass.getpass("Password: ")
        if getpass.getpass("Repeat password: ") != password:
            print("Passwords do not match.", file=sys.stderr)
            return 1

    try:
        data = RegisterRequest(email=args.email, name=args.name or args.email.split("@")[0], password=password)
    except ValidationError as exc:
        for err in exc.errors():
            print(f"{err['loc'][0]}: {err['msg']}", file=sys.stderr)
        return 1

    init_db()
    with get_session_factory()() as db:
        if db.query(models.User).filter(models.User.email == data.email).first() is not None:
            print(f"A user with email {data.email} already exists.", file=sys.stderr)
            return 1
        db.add(models.User(email=data.email, name=data.name, password_hash=hash_password(data.password)))
        db.commit()
    print(f"Created user {data.email}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
