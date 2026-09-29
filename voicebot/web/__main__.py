"""Dashboard admin from the server's shell (works while the bot runs; it picks the change up):

    .venv/bin/python -m voicebot.web set-password     # create/replace the local admin login
    .venv/bin/python -m voicebot.web signout-all      # invalidate every dashboard session
"""
import getpass
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])  # data/dashboard.json is relative to the project

from .auth import AuthStore  # noqa: E402


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    store = AuthStore()
    if cmd == "set-password":
        user = input(f"Username [{store.password_user or 'admin'}]: ").strip() or store.password_user or "admin"
        pw = getpass.getpass("New password (10+ characters): ")
        if len(pw) < 10:
            print("Too short.")
            return 1
        if getpass.getpass("Again: ") != pw:
            print("Passwords don't match.")
            return 1
        store.set_password(user, pw)
        print(f"Password login set for '{user}'. Other password sessions were signed out.")
        return 0
    if cmd == "signout-all":
        store.sign_out_everyone()
        print("Every dashboard session is signed out.")
        return 0
    print(__doc__)
    return 1


sys.exit(main())
