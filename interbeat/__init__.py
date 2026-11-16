"""interbeat - cross-application beat linking prototype for interactive sound.

Independent sequencer apps share tempo, beat phase and start/stop state over
UDP. Each app renders real audio from its local clock mapped onto the shared
beat grid through a de-jittered, slew-limited beat map.
"""

__version__ = "0.1.0"
