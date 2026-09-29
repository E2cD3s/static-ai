"""Static on Fluxer (platform.mode: fluxer / both). See bot.py.

FluxerBot is imported lazily, so light modules here (help.py) can be used without loading fluxer.py/LiveKit."""


def __getattr__(name):
    if name == "FluxerBot":
        from .bot import FluxerBot
        return FluxerBot
    raise AttributeError(name)


__all__ = ["FluxerBot"]
