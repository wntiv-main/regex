"""Example match explorer GUI for email regex"""

import sys
import tkinter
from tkinter import Tk, Text

from regex import Regex

# The regex to search for
email_rx = Regex(r"(\w+(?:\.\w+)*)@(\w+(?:\.\w+)+)")
print("Loaded email regex")
phone_rx = Regex(r"(?:\+\d{1,2}\s*)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}")
print("Loaded phone regex")

# For pyinstaller build testing
if "--just-testing" in sys.argv:
    sys.exit()

win = Tk()
text_field = Text(win)
text_field.pack()
text_field.insert(tkinter.END, "Example text")
# TKinter tags work kinda like <span>s with CSS styles
text_field.tag_config("email_match", background="lime")
text_field.tag_config("phone_match", background="yellow")

def row_col_coords(string: str, idx: int):
    """
    Converts an index into a string into TKinter's row/column indexing

    Arguments:
        string -- The string being indexed into
        idx -- The index into the string

    Returns:
        The string representation of the row/column coord, as used by
        TKinter
    """
    # "Trust me bro"
    return (f"{string[:idx].count('\n') + 1}."
            f"{string[idx - 1::-1].find('\n') % (idx + 1)}")

def tag_remove_all(widget, tag_name):
    """
    Remove all instances of a tag from the given widget

    Arguments:
        widget -- The Text widget to remove the tags from
        tag_name -- The tag to remove all occurences of
    """
    # "borrowed" from the web - see:
    # https://stackoverflow.com/a/23295120/13160456
    ranges = list(map(str, text_field.tag_ranges(tag_name)))
    for s, e in zip(ranges[::2], ranges[1::2]):
        widget.tag_remove(tag_name, s, e)

def on_change(*_):
    """Keypress event listener to update highlights"""
    text = text_field.get("1.0", tkinter.END)
    # Remove old tags
    tag_remove_all(text_field, "email_match")
    tag_remove_all(text_field, "phone_match")
    # Add updated tags
    for idx in email_rx.match(text):
        text_field.tag_add('email_match',
                           row_col_coords(text, idx[0].start),
                           row_col_coords(text, idx[0].stop))
    for idx in phone_rx.match(text):
        text_field.tag_add('phone_match',
                           row_col_coords(text, idx[0].start),
                           row_col_coords(text, idx[0].stop))

# Prepare initial highlights
on_change()

text_field.bind("<KeyRelease>", on_change)

win.mainloop()
