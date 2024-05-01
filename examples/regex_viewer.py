"""Example match explorer GUI for email regex"""

# pylint: disable=all
from regex import Regex
import tkinter
from tkinter import Tk, Text

# The regex to search for
rx = Regex(r"(\w+(?:\.\w+)*)@(\w+(?:\.\w+)+)")

win = Tk()
text_field = Text(win)
text_field.pack()
text_field.insert(tkinter.END, "Example text")
text_field.tag_config("match", background="yellow")

def row_col_coords(string: str, idx: int):
    # "Trust me bro"
    return (f"{string[:idx].count('\n') + 1}."
            f"{string[idx - 1::-1].find('\n') % (idx + 1)}")

def tag_remove_all(widget, tag_name):
    ranges = list(map(str, text_field.tag_ranges(tag_name)))
    for s, e in zip(ranges[::2], ranges[1::2]):
        widget.tag_remove(tag_name, s, e)

def on_change(*args):
    text = text_field.get("1.0", tkinter.END)
    tag_remove_all(text_field, "match")
    for idx in rx.match(text):
        text_field.tag_add('match',
                  row_col_coords(text, idx[0].start),
                  row_col_coords(text, idx[0].stop))

text_field.bind("<KeyRelease>", on_change)

win.mainloop()
