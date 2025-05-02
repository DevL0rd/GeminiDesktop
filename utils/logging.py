import ctypes
import random
import re
import os
from time import gmtime, strftime
import time
from utils.events import Events


class Logger:
    Events = None
    prefix = "No prefix"
    prefixColor = None
    timeColor = "blue"
    useColor = True
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    EXCEPTION = 4
    NONE = 5
    logging_level = 1
    formatting = {
        "reset": '\033[0m',
        "bold": '\033[01m',
        "disable": '\033[02m',
        "underline": '\033[04m',
        "reverse": '\033[07m',
        "strikethrough": '\033[09m',
        "invisible": '\033[08m',
        "fg": {
            'white':    "\033[1;37m",
            'yellow':   "\033[1;33m",
            'green':    "\033[1;32m",
            'blue':     "\033[1;34m",
            'cyan':     "\033[1;36m",
            "pink": '\033[1;95m',
            'red':      "\033[1;31m",
            'purple':  "\033[1;35m",
            'grey':  "\033[0;37m",
            'gray':  "\033[0;37m",
            "darkGrey": '\033[1;90m',
            "darkGray": '\033[1;90m',
            'darkYellow': "\033[0;33m",
            'darkGreen':  "\033[0;32m",
            'darkBlue':   "\033[0;34m",
            'darkCyan':   "\033[0;36m",
            'darkRed':    "\033[0;31m",
            'darkPurple': "\033[0;35m",
            'black':  "\033[0;30m"
        },
        "bg": {
            "black": '\033[40m',
            "red": '\033[41m',
            "green": '\033[42m',
            "yellow": '\033[43m',
            "blue": '\033[44m',
            "purple": '\033[45m',
            "cyan": '\033[46m',
            "lightGrey": '\033[47m',
            "lightGray": '\033[47m'
        }
    }

    def getRealTextLength(self, text):
        for code in self.formatting["fg"]:
            text = text.replace(self.formatting["fg"][code], "")
        for code in self.formatting["bg"]:
            text = text.replace(self.formatting["bg"][code], "")
        return len(text)

    def getColoredTimeString(self):
        timeString = strftime("%H:%M:%S", gmtime())
        return self.formatting['fg']["white"] + \
            "[" + self.formatting['fg'][self.timeColor] + \
            timeString + self.formatting['fg']["white"] + "]"

    def getColoredprefixString(self, level):
        color = self.prefixColor
        if color is None:
            color = self.getLevelColor(level)

        return self.formatting['fg']["white"] + \
            "[" + color + \
            self.prefix + self.formatting['fg']["white"] + "]"

    def getLevelColor(self, level):
        levelColor = self.formatting['fg']["white"]
        if level == Logger.DEBUG:
            levelColor = self.formatting['fg']["darkGray"]
        elif level == Logger.INFO:
            levelColor = self.formatting['fg']["green"]
        elif level == Logger.WARN:
            levelColor = self.formatting['fg']["yellow"]
        elif level == Logger.ERROR:
            levelColor = self.formatting['fg']["red"]
        elif level == Logger.EXCEPTION:
            levelColor = self.formatting['fg']["red"]

        return levelColor

    def getColoredLevelString(self, level):
        levelString = "NONE"
        if level == Logger.DEBUG:
            levelString = "DEBUG"
        elif level == Logger.INFO:
            levelString = "INFO"
        elif level == Logger.WARN:
            levelString = "WARN"
        elif level == Logger.ERROR:
            levelString = "ERROR"
        elif level == Logger.EXCEPTION:
            levelString = "EXCEPTION"
        color = self.getLevelColor(level)

        # uhh makes sense sure
        return self.formatting['fg']["white"] + "[" + color + levelString + self.formatting['fg']["white"] + "]"

    def getColoredLogPrefixes(self, level):
        timeStringColored = self.getColoredTimeString()
        prefixString = self.getColoredprefixString(level)
        levelString = self.getColoredLevelString(level)
        return timeStringColored + "-" + levelString + "-" + prefixString + ": "

    def reColorQuotedStrings(self, text, textColor, highlightColor="blue"):
        strings = re.findall(r"'(.*?)'", text)
        for string in strings:
            text = text.replace(
                string, self.formatting["fg"][highlightColor] + string + self.formatting["fg"][textColor])
        return text

    def formatAndColorLogText(self, text, color, level):
        text = self.formatting['fg'][color] + text + self.formatting['reset']
        coloredLogPrefixes = self.getColoredLogPrefixes(level)
        formatedLogText = self.reColorQuotedStrings(text, color)
        return coloredLogPrefixes + formatedLogText

    def getTimeString(self):
        timeString = strftime("%H:%M:%S", gmtime())
        return timeString

    def getprefixString(self):
        return self.prefix

    def getLevelString(self, level):
        levelString = "NONE"
        if level == Logger.DEBUG:
            levelString = "DEBUG"
        elif level == Logger.INFO:
            levelString = "INFO"
        elif level == Logger.WARN:
            levelString = "WARN"
        elif level == Logger.ERROR:
            levelString = "ERROR"
        elif level == Logger.EXCEPTION:
            levelString = "CRITICAL"
        return levelString

    def getLogPrefixes(self, level):
        return "[" + self.getTimeString() + "]-[" + self.getLevelString(level) + "]-[" + self.getprefixString() + "]: "

    def formatLogText(self, text, level):
        return self.getLogPrefixes(level) + text

    def print(self, text, color="white"):
        colorText = f"{self.formatting['fg'][color]}{text}{self.formatting['reset']}"
        if self.useColor:
            print(colorText)
            Logger.trigger("print", colorText)
        else:
            print(text)
            Logger.trigger("print", text)

    def highlight(self, text, color="blue"):
        colorText = self.formatting["bg"][color] + "\n" + \
            self.formatting["bold"] + text + self.formatting["reset"] + "\n"
        normalText = self.formatLogText(text, level=Logger.DEBUG)
        if Logger.logging_level <= Logger.INFO:
            if self.useColor:
                print(colorText)
                Logger.trigger("print", colorText)
            else:
                print(normalText)
                Logger.trigger("print", normalText)

    def debug(self, text, color="darkGray"):
        colorText = self.formatAndColorLogText(text, color, Logger.DEBUG)
        normalText = self.formatLogText(text, Logger.DEBUG)
        if Logger.logging_level <= Logger.DEBUG:
            if self.useColor:
                print(colorText)
                Logger.trigger("print", colorText)
            else:
                print(normalText)
                Logger.trigger("print", normalText)

    def info(self, text, color="white"):
        colorText = self.formatAndColorLogText(text, color, Logger.INFO)
        normalText = self.formatLogText(text, Logger.INFO)
        if Logger.logging_level <= Logger.INFO:
            if self.useColor:
                print(colorText)
                Logger.trigger("print", colorText)
            else:
                print(normalText)
                Logger.trigger("print", normalText)

    def warn(self, text, color="darkYellow"):
        colorText = self.formatAndColorLogText(text, color, Logger.WARN)
        normalText = self.formatLogText(text, Logger.WARN)
        if Logger.logging_level <= Logger.WARN:
            if self.useColor:
                print(colorText)
                Logger.trigger("print", colorText)
            else:
                print(normalText)
                Logger.trigger("print", normalText)

    def error(self, text, color="red"):
        colorText = self.formatAndColorLogText(text, color, Logger.ERROR)
        normalText = self.formatLogText(text, Logger.ERROR)
        if Logger.logging_level <= Logger.ERROR:
            if self.useColor:
                print(colorText)
                Logger.trigger("print", colorText)
            else:
                print(normalText)
                Logger.trigger("print", normalText)

    def exception(self, text, color="red"):
        colorText = self.formatAndColorLogText(text, color, Logger.EXCEPTION)
        normalText = self.formatLogText(text, Logger.EXCEPTION)
        if Logger.logging_level <= Logger.EXCEPTION:
            if self.useColor:
                print(colorText)
                Logger.trigger("print", colorText)
            else:
                print(normalText)
                Logger.trigger("print", normalText)

    @staticmethod
    def clearScreen():
        Logger.trigger("print", "*CLEARSCREEN*")
        os.system('cls' if os.name == 'nt' else 'clear')

    def setPrefixColor(self, color):
        self.prefixColor = color

    def setPrefix(self, prefix, color="default"):
        self.prefix = prefix
        if color != "default":
            self.setPrefixColor(color)

    def setColorMode(self, useColor):
        self.useColor = useColor

    def setLevel(level):
        Logger.logging_level = level

    @staticmethod
    def printSplash():
        os.system('cls' if os.name == 'nt' else 'clear')
        art = """

  _____          _      _   ___          __   __          
 / ___/__ __ _  (_)__  (_) / _ \___ ___ / /__/ /____  ___ 
/ (_ / -_)  ' \/ / _ \/ / / // / -_|_-</  '_/ __/ _ \/ _ \\
\___/\__/_/_/_/_/_//_/_/ /____/\__/___/_/\_\\\\__/\___/ .__/
                                                   /_/    
"""
        print(Logger.formatting['fg']["blue"] +
              art + Logger.formatting['reset'])
        print(Logger.formatting['fg']["blue"] + "~ DevL0rd" +
              Logger.formatting['reset'])
        print()

    def __init__(self, prefix="No prefix", useColor=True):
        self.setPrefix(prefix)

        if not Logger.Events:
            Logger.Events = Events()
            Logger.on = Logger.Events.on
            Logger.trigger = Logger.Events.trigger

        self.setColorMode(useColor)
