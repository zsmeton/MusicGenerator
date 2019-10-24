import readline
from pip._vendor.distlib.compat import raw_input


def get_user_yes_no(message):
    """
    Prints a prompt and gets a yes or no response from the user
    :param message: the prompt to ask the user
    :return: bool
    """
    while True:
        usr_input = input("{} (y/n): ".format(message)).capitalize()
        if usr_input == "YES" or usr_input == "Y":
            return True
        elif usr_input == "NO" or usr_input == "N":
            return False
        else:
            print(usr_input, "is not a valid response please enter (y/n).")


def get_user_non_negative_number(message=None):
    """
    Prints a prompt and gets a number response from the user
    :param message: the prompt to ask the user
    :return: float
    """
    while True:
        if message is not None:
            usr_input = input(f"{message}: ").capitalize()
        else:
            usr_input = input().capitalize()
        if usr_input.isdecimal() and float(usr_input) >= 0:
            return float(usr_input)
        else:
            print(usr_input, "is not a valid number please try again.")


def get_user_non_negative_number_or_default(message, default_message='default', default_key='*',default_value=None):
    """
    Prints a prompt and gets a number response from the user
    :param message: the prompt to ask the user
    :param default_value:
    :param default_key:
    :param default_message:
    :return: float
    """
    while True:
        if default_message or default_key:
            usr_input = input(f"{message} or {default_key} for {default_message}: ").capitalize()
        else:
            usr_input = input(f"{message}: ").capitalize()
        if usr_input.isdecimal() and float(usr_input) >= 0:
            return float(usr_input)
        elif usr_input in default_key:
            return default_value
        else:
            print(usr_input, "is not a valid number please try again.")


def get_user_options(message, options):
    """
    Prints a prompt and gets the users choice between a few options
    :param message: the message to display before the options are listed
    :param options: the options to list
    :return: the index of the chosen option
    """
    while True:
        # Print message
        print("{} :".format(message), flush=True)
        options = [str(option).lower() for option in options]
        for i, option in enumerate(options):
            print("({}) {}".format(i+1, option), flush=True)
        # get user input
        usr_input = int(get_user_non_negative_number())
        if 1 <= usr_input <= len(options):
            return int(usr_input)
        else:
            print(usr_input, "is not a valid option please try again.", flush=True)


def get_user_filename(message):
    readline.parse_and_bind("tab: complete")
    readline.set_completer_delims(" \t")
    line = raw_input(f'{message}: ')
    readline.set_completer_delims(" ?>q")
    return line