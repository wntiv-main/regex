"""Example showing possible usecase: validating phone numbers"""

from regex import Regex


def main():
    """Entrypoint"""
    # Example phone number regex, adapted from:
    # https://stackoverflow.com/a/16699507/13160456
    email_regex = Regex(
        r"\A(?:\+\d{1,2}\s*)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\Z")

    # example matches:
    # 1234567890
    # 123-456-7890
    # (123) 456-7890
    # 123 456 7890
    # 123.456.7890
    # +91 (123) 456-7890

    while user_input := input("Please enter an phone number: "):
        print(
            "You have entered",
            "a valid" if email_regex.test(user_input) else "an invalid",
            "phone number")


if __name__ == "__main__":
    main()
