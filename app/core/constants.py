"""Application constants."""

from enum import Enum
from typing import Dict, List, Tuple


class MoodLabel(str, Enum):
    """Mood classification labels."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    BORED = "bored"
    CONFUSED = "confused"


# Mood descriptions for display
MOOD_DESCRIPTIONS: Dict[str, str] = {
    MoodLabel.HAPPY: "😊 Feeling joyful and positive",
    MoodLabel.SAD: "😢 Feeling down or melancholic",
    MoodLabel.ANGRY: "😠 Feeling frustrated or irritated",
    MoodLabel.ANXIOUS: "😰 Feeling worried or stressed",
    MoodLabel.NEUTRAL: "😐 Feeling balanced and calm",
    MoodLabel.EXCITED: "🤩 Feeling energetic and enthusiastic",
    MoodLabel.BORED: "😴 Feeling uninterested or dull",
    MoodLabel.CONFUSED: "😕 Feeling uncertain or puzzled",
}

# Color mapping for moods (for UI)
MOOD_COLORS: Dict[str, str] = {
    MoodLabel.HAPPY: "#FFD93D",
    MoodLabel.SAD: "#6C7A89",
    MoodLabel.ANGRY: "#E74C3C",
    MoodLabel.ANXIOUS: "#9B59B6",
    MoodLabel.NEUTRAL: "#95A5A6",
    MoodLabel.EXCITED: "#F39C12",
    MoodLabel.BORED: "#BDC3C7",
    MoodLabel.CONFUSED: "#3498DB",
}

# Sample training data for mood detection
SAMPLE_TRAINING_DATA: List[Tuple[str, str]] = [
    # Happy
    ("I am so happy today! Everything is going great.", MoodLabel.HAPPY),
    ("What a wonderful day! I feel fantastic.", MoodLabel.HAPPY),
    ("Life is beautiful and I'm smiling.", MoodLabel.HAPPY),
    ("So grateful for all the good things happening.", MoodLabel.HAPPY),
    ("Just got promoted! Best day ever!", MoodLabel.HAPPY),
    ("Spending time with family makes me so happy.", MoodLabel.HAPPY),
    ("The sun is shining and my heart is full.", MoodLabel.HAPPY),
    ("Achieved my goals! Feeling on top of the world.", MoodLabel.HAPPY),
    ("Love is in the air and I'm feeling great.", MoodLabel.HAPPY),
    ("Nothing can stop my good mood today!", MoodLabel.HAPPY),

    # Sad
    ("I feel so down today. Nothing seems right.", MoodLabel.SAD),
    ("Missing someone special hurts so much.", MoodLabel.SAD),
    ("The world feels gray and empty right now.", MoodLabel.SAD),
    ("Can't stop crying. Everything feels hopeless.", MoodLabel.SAD),
    ("Lost my job today. Feeling devastated.", MoodLabel.SAD),
    ("Sometimes life just isn't fair.", MoodLabel.SAD),
    ("Feeling lonely and misunderstood.", MoodLabel.SAD),
    ("The pain is too much to bear.", MoodLabel.SAD),
    ("Nothing brings me joy anymore.", MoodLabel.SAD),
    ("Tears keep falling and I don't know why.", MoodLabel.SAD),

    # Angry
    ("I am so furious right now! This is unacceptable!", MoodLabel.ANGRY),
    ("Why does everything have to be so difficult!", MoodLabel.ANGRY),
    ("I can't believe they would do this to me!", MoodLabel.ANGRY),
    ("So tired of dealing with incompetence.", MoodLabel.ANGRY),
    ("My blood is boiling with rage!", MoodLabel.ANGRY),
    ("This injustice makes me want to scream!", MoodLabel.ANGRY),
    ("I hate when people don't respect my time.", MoodLabel.ANGRY),
    ("Frustrated beyond words right now.", MoodLabel.ANGRY),
    ("Ready to explode with anger!", MoodLabel.ANGRY),
    ("This is the last straw! I've had enough!", MoodLabel.ANGRY),

    # Anxious
    ("My heart is racing and I can't breathe.", MoodLabel.ANXIOUS),
    ("So worried about what might happen tomorrow.", MoodLabel.ANXIOUS),
    ("Can't sleep, mind won't stop racing.", MoodLabel.ANXIOUS),
    ("Feeling overwhelmed and stressed out.", MoodLabel.ANXIOUS),
    ("What if everything goes wrong?", MoodLabel.ANXIOUS),
    ("Panic attack coming on, need to calm down.", MoodLabel.ANXIOUS),
    ("Too many things to worry about.", MoodLabel.ANXIOUS),
    ("Constantly on edge, can't relax.", MoodLabel.ANXIOUS),
    ("The pressure is crushing me.", MoodLabel.ANXIOUS),
    ("Dreading what comes next.", MoodLabel.ANXIOUS),

    # Neutral
    ("It's just another regular day.", MoodLabel.NEUTRAL),
    ("Nothing special happening today.", MoodLabel.NEUTRAL),
    ("Feeling pretty average right now.", MoodLabel.NEUTRAL),
    ("Just going through the motions.", MoodLabel.NEUTRAL),
    ("Today is neither good nor bad.", MoodLabel.NEUTRAL),
    ("Taking things as they come.", MoodLabel.NEUTRAL),
    ("Maintaining a steady pace in life.", MoodLabel.NEUTRAL),
    ("Everything is proceeding normally.", MoodLabel.NEUTRAL),
    ("Just another day at work.", MoodLabel.NEUTRAL),
    ("No strong feelings one way or another.", MoodLabel.NEUTRAL),

    # Excited
    ("I can't wait for the weekend! So excited!", MoodLabel.EXCITED),
    ("My heart is pounding with excitement!", MoodLabel.EXCITED),
    ("Counting down the hours until the concert!", MoodLabel.EXCITED),
    ("Adrenaline rushing through my veins!", MoodLabel.EXCITED),
    ("Going on vacation tomorrow! Woohoo!", MoodLabel.EXCITED),
    ("Big presentation today, feeling pumped!", MoodLabel.EXCITED),
    ("New beginnings are so thrilling!", MoodLabel.EXCITED),
    ("Can't contain my excitement any longer!", MoodLabel.EXCITED),
    ("Something amazing is about to happen!", MoodLabel.EXCITED),
    ("The anticipation is killing me!", MoodLabel.EXCITED),

    # Bored
    ("Nothing to do, just sitting around.", MoodLabel.BORED),
    ("Time is moving so slowly today.", MoodLabel.BORED),
    ("Wish something interesting would happen.", MoodLabel.BORED),
    ("Scrolling through the same apps repeatedly.", MoodLabel.BORED),
    ("Can't find anything entertaining to do.", MoodLabel.BORED),
    ("Everything feels so mundane.", MoodLabel.BORED),
    ("Just watching the clock tick by.", MoodLabel.BORED),
    ("Life feels like a repetitive routine.", MoodLabel.BORED),
    ("Seeking some excitement but finding none.", MoodLabel.BORED),
    ("So tired of doing the same old thing.", MoodLabel.BORED),

    # Confused
    ("I don't understand what's happening here.", MoodLabel.CONFUSED),
    ("So many questions, no clear answers.", MoodLabel.CONFUSED),
    ("This doesn't make any sense to me.", MoodLabel.CONFUSED),
    ("Feeling lost and disoriented.", MoodLabel.CONFUSED),
    ("Can't figure out what to do next.", MoodLabel.CONFUSED),
    ("Everything seems contradictory right now.", MoodLabel.CONFUSED),
    ("Where am I and what am I doing?", MoodLabel.CONFUSED),
    ("Need help understanding this situation.", MoodLabel.CONFUSED),
    ("Too much conflicting information.", MoodLabel.CONFUSED),
    ("Why is everything so complicated?", MoodLabel.CONFUSED),
]

# Content recommendations by mood
MOOD_RECOMMENDATIONS: Dict[str, List[Dict[str, str]]] = {
    MoodLabel.HAPPY: [
        {"type": "music", "title": "Happy - Pharrell Williams", "url": "https://open.spotify.com/track/60nZcImufyMAFUHKzRcysG"},
        {"type": "music", "title": "Don't Stop Me Now - Queen", "url": "https://open.spotify.com/track/7hQJA50XrCWABAu5vVX8tP"},
        {"type": "activity", "title": "Share your joy with friends", "description": "Call someone you love and share your good news"},
        {"type": "movie", "title": "The Secret Life of Walter Mitty", "description": "An inspiring adventure film"},
        {"type": "quote", "content": """Happiness is not something ready-made. It comes from your own actions."" - Dalai Lama"""},
    ],
    MoodLabel.SAD: [
        {"type": "music", "title": "Fix You - Coldplay", "url": "https://open.spotify.com/track/7LVHVU3tWfcxj5aiPFEW4Q"},
        {"type": "music", "title": "Someone Like You - Adele", "url": "https://open.spotify.com/track/4kflIGfjdZJW4ot2ioixTB"},
        {"type": "activity", "title": "Practice self-compassion", "description": "Take a warm bath and be kind to yourself"},
        {"type": "movie", "title": "Inside Out", "description": "A beautiful exploration of emotions"},
        {"type": "quote", "content": """It's okay not to be okay. This too shall pass."" - Unknown"""},
    ],
    MoodLabel.ANGRY: [
        {"type": "music", "title": "Break Stuff - Limp Bizkit", "url": "https://open.spotify.com/track/0YSEJ0mV0y8PQN3eKUKb48"},
        {"type": "music", "title": "Given Up - Linkin Park", "url": "https://open.spotify.com/track/3r9gMP8hD3G7cR9F5P7j3V"},
        {"type": "activity", "title": "Channel anger into exercise", "description": "Go for a run or hit a punching bag"},
        {"type": "movie", "title": "The Equalizer", "description": "Action film for cathartic release"},
        {"type": "quote", "content": """For every minute you remain angry, you give up sixty seconds of peace of mind."" - Ralph Waldo Emerson"""},
    ],
    MoodLabel.ANXIOUS: [
        {"type": "music", "title": "Weightless - Marconi Union", "url": "https://open.spotify.com/track/6Jz3ce3"},
        {"type": "music", "title": "Clair de Lune - Debussy", "url": "https://open.spotify.com/track/6Jz3ce3"},
        {"type": "activity", "title": "Try the 4-7-8 breathing technique", "description": "Inhale 4s, hold 7s, exhale 8s"},
        {"type": "movie", "title": "My Neighbor Totoro", "description": "Gentle, soothing Studio Ghibli film"},
        {"type": "quote", "content": """This too shall pass. Breathe in courage, exhale fear."" - Unknown"""},
    ],
    MoodLabel.NEUTRAL: [
        {"type": "music", "title": "Weightless - Marconi Union", "url": "https://open.spotify.com/track/6Jz3ce3"},
        {"type": "music", "title": "Lo-fi Beats to Study/Relax", "url": "https://open.spotify.com/playlist/0vvXsWCC9xrXsKd4Fy0"},
        {"type": "activity", "title": "Mindfulness meditation", "description": "10 minutes of present-moment awareness"},
        {"type": "movie", "title": "The Grand Budapest Hotel", "description": "Beautifully balanced Wes Anderson film"},
        {"type": "quote", "content": """Peace comes from within. Do not seek it without."" - Buddha"""},
    ],
    MoodLabel.EXCITED: [
        {"type": "music", "title": "Can't Hold Us - Macklemore", "url": "https://open.spotify.com/track/1mKzBj8bFJK"},
        {"type": "music", "title": "Titanium - David Guetta ft. Sia", "url": "https://open.spotify.com/track/4y1LsJ1M"},
        {"type": "activity", "title": "Document your excitement", "description": "Journal about what you're looking forward to"},
        {"type": "movie", "title": "Mad Max: Fury Road", "description": "High-octane thrill ride"},
        {"type": "quote", "content": """The only way to do great work is to love what you do."" - Steve Jobs"""},
    ],
    MoodLabel.BORED: [
        {"type": "music", "title": "The Less I Know The Better - Tame Impala", "url": "https://open.spotify.com/track/4gF"},
        {"type": "music", "title": "Do I Wanna Know? - Arctic Monkeys", "url": "https://open.spotify.com/track/5K"},
        {"type": "activity", "title": "Learn something new", "description": "Try a free online course on something random"},
        {"type": "movie", "title": "The Matrix", "description": "Mind-bending sci-fi classic"},
        {"type": "quote", "content": """Boredom is the root of all evil. The despairing refusal to be oneself."" - Søren Kierkegaard"""},
    ],
    MoodLabel.CONFUSED: [
        {"type": "music", "title": "The Scientist - Coldplay", "url": "https://open.spotify.com/track/75JFxkwe"},
        {"type": "music", "title": "Mad World - Gary Jules", "url": "https://open.spotify.com/track/3b2"},
        {"type": "activity", "title": "Break it down", "description": "Write down what's confusing you and tackle one piece at a time"},
        {"type": "movie", "title": "Inception", "description": "Complex but ultimately rewarding"},
        {"type": "quote", "content": """The important thing is not to stop questioning."" - Albert Einstein"""},
    ],
}

# API response messages
API_MESSAGES = {
    "HEALTH_OK": "MoodSense AI is running smoothly",
    "PREDICTION_SUCCESS": "Mood prediction completed successfully",
    "INVALID_INPUT": "Invalid input text provided",
    "MODEL_ERROR": "Error occurred during model prediction",
    "SERVICE_UNAVAILABLE": "Service temporarily unavailable",
}
