# Data Directory

Place prepared audio clips here, organised by health grade:

```
data/
├── normal/     # Grade 0 — healthy engine
├── warning/    # Grade 1 — early degradation
├── fault/      # Grade 2 — confirmed fault
└── critical/   # Grade 3 — severe / imminent failure
```

Accepted formats: `.wav` `.mp3` `.m4a` `.flac` `.ogg`

To prepare recordings from raw audio files:
```bash
python data_preparation/car_engine_prepare.py
```
