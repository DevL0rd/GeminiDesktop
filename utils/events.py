#Author: https://github.com/DevL0rd
import uuid


class Events:
    def __init__(self):
        self.events = {
            "event": {},
            "unhandledEvent": {}
        }

    def on(self, eventName: str, callBack: callable):
        if eventName not in self.events:
            self.events[eventName] = {}
        newUUID = str(uuid.uuid4())
        self.events[eventName][newUUID] = callBack
        return newUUID

    def removeListener(self, eventName: str, callbackID: str):
        if eventName not in self.events:
            return
        eventCallbacks = self.events[eventName]
        if callbackID not in eventCallbacks:
            return
        del eventCallbacks[callbackID]

    def trigger(self, eventName: str, data={}):
        for eventCallbackId in self.events["event"]:
            # Emit a genereal event event to pass all events to something.
            eventCallback = self.events["event"][eventCallbackId]
            eventCallback(eventName, data)

        if eventName in self.events:
            for eventCallbackId in self.events[eventName]:
                eventCallback = self.events[eventName][eventCallbackId]
                if eventCallback(data):
                    return  # if the event returns true, stop processing events as it was handled
            return

        for eventCallbackId in self.events["unhandledEvent"]:
            # If it was unhandled emit a unhandled event
            eventCallback = self.events["unhandledEvent"][eventCallbackId]
            if eventCallback(eventName, data):
                return  # if the event returns true, stop processing events as it was handled
