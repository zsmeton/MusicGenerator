def get_user_yes_no(message):
    """
    Prints a prompt and gets a yes or no response from the user
    :param message: the prompt to ask the user
    :return: bool
    """
    while True:
        usr_input = input("{} (y/n)".format(message)).capitalize()
        if usr_input == "YES" or usr_input == "Y":
            return True
        elif usr_input == "NO" or usr_input == "N":
            return False
        else:
            print(usr_input, "is not a valid response please enter (y/n).")


def get_user_non_negative_number(message):
    """
    Prints a prompt and gets a number response from the user
    :param message: the prompt to ask the user
    :return: float
    """
    while True:
        usr_input = input("{}".format(message)).capitalize()
        if usr_input.isdecimal() and float(usr_input) >= 0:
            return float(usr_input)
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
        print("{} :".format(message))
        options = [str(option).lower() for option in options]
        for i, option in enumerate(options):
            print("({}) {}".format(i, option))
        # get user input
        usr_input = input()
        if int(usr_input) >= 0 and int(usr_input) < len(options):
            return int(usr_input)
        else:
            print(usr_input, "is not a valid option please try again.")