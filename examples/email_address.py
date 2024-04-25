"""Example showing possible usecase: validating email addresses"""

from regex import Regex


def main():
    """Entrypoint"""
    # Example (NOT accurate) email regex
    # Should match e.g: w0rd.more_w0rds@s0meth1ng.s0meth1ng.e1se
    email_regex = Regex(
        r"\A\w+(?:\.\w+)*@\w+(?:\.\w+)+\Z")

    while user_input := input("Please enter an email: "):
        print(
            "You have entered",
            "a valid" if email_regex.test(user_input) else "an invalid",
            "email address")


if __name__ == "__main__":
    main()
