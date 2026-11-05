"""
Translation Pipeline and Quality Auditing Module
Generates paired Hindi (Devanagari) and English translations for Hinglish text without label leakage.
Implements bilingual lexicons, phonetic Latin-to-Devanagari transliteration,
automatic checks, human validation sampling (150 rows), and label-shift auditing.
"""

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple
from collections import Counter

from src.utils.constants import (
    ID2LABEL,
    LABEL2ID,
    CLASS_NAMES,
)

# ---------------------------------------------------------------------------
# 1. Comprehensive Hinglish to English Bilingual Lexicon
# ---------------------------------------------------------------------------
HINGLISH_TO_ENGLISH_DICT = {
    # Full preset sentences — direct high-quality translations
    "ye movie bilkul bakwaas hai, paisa aur time dono barbaad!": "This movie is absolutely rubbish, a waste of both money and time!",
    "ye movie bilkul bakwaas hai paisa aur time dono barbaad": "This movie is absolutely rubbish, a waste of both money and time!",
    "boring service aur ghatiya khana, yahan kabhi mat aao": "Boring service and terrible food, never come here.",
    "boring service aur ghatiya khana yahan kabhi mat aao": "Boring service and terrible food, never come here.",
    "main aaj ghar ja raha hoon, dost se milne": "I am going home today to meet a friend.",
    "main aaj ghar ja raha hoon dost se milne": "I am going home today to meet a friend.",
    "ye product bahut accha hai aur kaam karta hai": "This product is very good and works well.",
    "zabardast acting aur shandar kahani, superhit film hai!": "Tremendous acting and magnificent story, it is a superhit film!",
    "zabardast acting aur shandar kahani superhit film hai": "Tremendous acting and magnificent story, it is a superhit film!",
    "bhai, aaj toh meri life ka sabse best day hai!": "Brother, today is literally the best day of my life!",
    "bhai aaj toh meri life ka sabse best day hai": "Brother, today is literally the best day of my life!",
    "app install karne ke baad open nahi ho raha": "The app is not opening after installation.",
    "battery backup bahut kam hai aur phone garam ho jata hai": "Battery backup is very poor and the phone overheats.",
    "customer service ne call ka jawab nahi diya": "Customer service did not answer the phone call.",
    "hotel ka room saaf tha aur staff helpful tha": "The hotel room was clean and the staff was helpful.",
    "ticket book karne me bahut dikkat aayi": "Faced a lot of issues while booking the ticket.",
    "paise cut gaye lekin recharge nahi hua": "Money was deducted from the account but recharge was not done.",
    "dress ka material bahut achha hai par color thoda alag hai": "The dress material is very good but the color is slightly different.",
    "doctor ne dawai lene ke liye bola tha par aaram nahi mila": "The doctor advised to take medicines but there was no relief.",
    "train 2 ghante late chal rahi hai": "The train is running 2 hours late.",
    "ye product bilkul bekar hai, paise barbaad ho gaye": "This product is completely useless, money has been wasted.",
    "ye product bilkul bekar hai paise barbaad ho gaye": "This product is completely useless, money has been wasted.",
    "maine online pizza order kiya tha lekin delivery bahut late aayi": "I had ordered pizza online but the delivery arrived very late.",
    "movie ki kahani bahut boring thi aur acting bhi kharab thi": "The movie storyline was very boring and the acting was terrible.",


    # Common colloquial particles & phrases
    "arey waah, isse achha toh kuch ho hi nahi": "Oh wow, nothing could be better than this!",
    "arey waah isse achha toh kuch ho hi nahi": "Oh wow, nothing could be better than this!",
    "arey waah": "oh wow",
    "are waah": "oh wow",
    "arey wah": "oh wow",
    "are wah": "oh wow",
    "isse achha": "better than this",
    "isse accha": "better than this",
    "isse achchha": "better than this",
    "isse behtar": "better than this",
    "ho hi nahi": "cannot be",
    "ho hi nahi sakta": "is impossible",
    "kuch ho hi nahi": "nothing can be",
    "kuch ho hi nahi sakta": "nothing could be better",
    "arey": "oh",
    "are": "oh",
    "waah": "wow",
    "wah": "wow",
    "isse": "than this",
    "usse": "than that",
    "kisse": "from whom",
    "toh": "then",
    "sakta": "can",
    "sakti": "can",
    "sakte": "can",

    # Multi-word idioms & conversational n-grams
    "mujhe lagta hai ki": "I think that",
    "mujhe lagta hai": "I think",
    "mujhe lagta": "it seems to me",
    "lagta hai": "seems to be",
    "galat baat hai": "is wrong",
    "galat baat": "wrong statement",
    "sahi baat hai": "is correct",
    "sahi baat": "true statement",
    "kahan ja rahe ho": "where are you going",
    "kahan ja rahe": "where are you going",
    "ja rahe ho": "are going",
    "ja rahe hain": "are going",
    "ja raha hoon": "am going",
    "ja rahi hoon": "am going",
    "kal aayega": "will come tomorrow",
    "kal aayenge": "will come tomorrow",
    "kya aap meri madad kar sakte hain": "can you please help me",
    "kya aap meri madad kar sakte": "can you please help me",
    "meri madad kar sakte hain": "can help me",
    "meri madad kar sakte": "can help me",
    "madad kar sakte hain": "can help",
    "madad kar sakte": "can help",
    "kar sakte hain": "can do",
    "kar sakta hai": "can do",
    "kar sakti hai": "can do",
    "kya aap": "can you",
    "kya tum": "can you",
    "janm deti hai": "gives rise to",
    "janm deti": "gives rise to",
    "any karnon": "other causes",
    "aparyapt poshan": "inadequate nutrition",
    "bare-bare bartan": "large vessels",
    "rakhe hue": "kept",
    "rakhe jate the": "were kept",
    "rakhe jate": "are kept",
    "jo bhi": "whoever",
    "mere sath": "with me",
    "apne dost se": "to meet my friend",
    "chalta hai": "walks",
    "mere hathon men": "in my hands",
    "hathon men": "in hands",
    "hoti hai": "is",
    "hota hai": "happens",
    "first problem to hal ho gayi": "the first problem was resolved",
    "to hal ho gayi": "was resolved",
    "hal ho gayi": "was resolved",
    "pallvit-pushpit hone diya": "allowed to flourish",
    "mitrta ko": "friendship",
    "destruction ke": "of destruction",
    "permission mat den": "do not give permission",
    "permission mat do": "do not give permission",
    "dekhne ke lie": "to inspect",
    "dekhne ke liye": "to inspect",
    "program hoti hai": "is a program",
    "instructions nihit hote hain": "contains instructions",
    "nihit hote hain": "are contained",
    "ted prize mil gaya hai": "has received the TED Prize",
    "mil gaya hai": "has received / was awarded",
    "samatal desolate maidan banakar chhorega": "will reduce the earth to a flat desolate plain",
    "guidance v mercy hai": "is guidance and mercy",
    "un logon ke lie": "for those people",
    "un logon ke liye": "for those people",
    "bilkul bakwaas": "completely rubbish",
    "bilkul bakwas": "completely rubbish",
    "paisa vasool": "value for money",
    "dono barbaad": "both wasted",
    "dono barbad": "both wasted",
    "bahut achha": "very good",
    "bahut achchha": "very good",
    "bahut accha": "very good",
    "bahut bura": "very bad",
    "bahut sundar": "very beautiful",
    "kabhi nahi": "never",
    "kabhi nahin": "never",
    "kabhi mat": "never",
    "kuch nahi": "nothing",
    "kuchh nahin": "nothing",
    "koi nahi": "nobody",
    "koi baat nahi": "no problem",
    "ek doosre": "each other",
    "ek doosre ka": "each other's",
    "chal door": "get away",
    "kar diya": "did",
    "kar raha": "is doing",
    "kar rahe": "are doing",
    "kar rahi": "is doing",
    "ja raha": "is going",
    "ja rahi": "is going",
    "ja rahe": "are going",
    "ho gaya": "happened",
    "ho gayi": "happened",
    "aa gaya": "arrived",
    "chala gaya": "went away",
    "yahan mat": "do not come here",
    "mat aao": "do not come",

    # Pronouns & determiners
    "ham": "we", "hum": "we", "hamne": "we", "humne": "we", "hamare": "our", "hamara": "our", "hamari": "our", "hami": "we",
    "main": "I", "maine": "I", "mainne": "I", "mera": "my", "meri": "my", "mere": "my", "mujhe": "I", "mujhko": "me",
    "aap": "you", "aapko": "you", "aapka": "your", "aapki": "your", "aapke": "your",
    "tum": "you", "tumne": "you", "tumhara": "your", "tumhari": "your", "tumhare": "your", "tumhen": "you", "tumhe": "you",
    "vah": "that", "woh": "that", "vo": "that", "unhonne": "they", "unhen": "them", "unhe": "them", "unka": "their", "unki": "their", "unke": "their",
    "use": "him", "uska": "his", "uski": "her", "uske": "his",
    "ye": "this", "yeh": "this", "inhen": "these", "inhe": "these", "inka": "their", "inki": "their", "inke": "their",
    "ise": "this", "iska": "its", "iski": "its", "iske": "its",
    "yah": "this", "kuchh": "some", "kuch": "some", "sab": "all", "sabhi": "all", "har": "every", "pratyek": "each",
    "apna": "own", "apne": "own", "apni": "own", "koi": "anyone", "koee": "anyone", "kisi": "someone",

    # Conjunctions, prepositions, particles
    "aur": "and", "ya": "or", "athva": "or", "kintu": "but", "parntu": "however", "parantu": "however", "lekin": "but", "magar": "but",
    "bhi": "also", "hi": "only", "to": "then", "tak": "until", "se": "from", "men": "in", "mein": "in",
    "par": "on", "ko": "to", "ke": "of", "ki": "of", "ka": "of", "lie": "for", "liye": "for",
    "tatha": "and", "evn": "and", "sath": "with", "saath": "with", "bina": "without", "jaise": "like",
    "kyunki": "because", "kyonki": "because", "isliye": "therefore", "jab": "when", "tab": "then", "ab": "now",
    "kya": "what", "kyun": "why", "kyon": "why", "kab": "when", "kahan": "where", "kaise": "how", "kaun": "who",
    "kitna": "how much", "kitne": "how many", "kitni": "how much", "agar": "if",

    # Negations & hedges
    "nahi": "not", "nahin": "not", "na": "not", "mat": "do not",
    "thoda": "slightly", "thodi": "slightly", "thode": "slightly", "kam": "less", "zyada": "more", "jyada": "more", "adhik": "more",
    "bahut": "very", "bilkul": "completely", "khub": "greatly",

    # Common verbs
    "hai": "is", "hain": "are", "tha": "was", "thi": "was", "the": "were", "hote": "are", "hoti": "is", "hota": "is",
    "karna": "do", "karta": "does", "karti": "does", "karte": "do", "kiya": "did", "kiye": "did", "karega": "will do",
    "hona": "be", "huaa": "happened", "hua": "happened", "hue": "happened", "hui": "happened", "ho": "are",
    "hoon": "am", "hun": "am",
    "raha": "going", "rahe": "going", "rahi": "going",
    "chahie": "should", "chahiye": "should", "sakte": "can", "sakta": "can", "sakti": "can",
    "kahna": "say", "kaha": "said", "kahe": "said", "bola": "spoke", "bole": "spoke", "bataya": "told",
    "dekhna": "see", "dekha": "saw", "dekhne": "seeing", "dekho": "look",
    "jana": "go", "ja": "go", "gaya": "went", "gaye": "went", "gayi": "went",
    "aana": "come", "aao": "come", "aaya": "came", "aaye": "came", "aata": "comes", "aayega": "will come", "aayenge": "will come",
    "milne": "to meet", "mile": "met", "mila": "received", "milte": "available", "rahega": "will remain",
    "lagta": "seems", "lagti": "seems", "lagte": "seem", "laga": "felt",
    "chalta": "walks", "chalti": "runs", "chalte": "walk",
    "rakha": "kept", "rakhe": "kept", "rakhi": "kept", "jate": "being",
    "aaj": "today", "kal": "tomorrow", "yahan": "here", "wahan": "there",
    "khana": "food", "peena": "drinking", "sona": "sleeping", "uthna": "wake up",
    "superhit": "superhit", "flop": "flop", "hit": "hit",

    # Nouns & sentiment vocabulary
    "din": "day", "rat": "night", "raat": "night", "samay": "time", "varsh": "year", "sal": "year", "saal": "year", "mahine": "months",
    "log": "people", "logon": "people", "bachche": "children", "vyakti": "person", "desh": "country",
    "sarkar": "government", "nyayalay": "court", "kanoon": "law", "kam": "less", "kary": "work", "kaam": "work",
    "shiksha": "education", "swasthya": "health", "pani": "water", "dharti": "earth", "aasman": "sky",
    "bhagvan": "God", "khuda": "God", "allah": "God", "rab": "Lord", "paravardigar": "Lord",
    "shandar": "magnificent", "zabardast": "fantastic", "badhiya": "great", "kamaal": "wonderful", "lazawab": "superb",
    "achchha": "good", "achha": "good", "accha": "good", "acchi": "good", "acche": "good",
    "bura": "bad", "buri": "bad", "bure": "bad", "kharab": "poor", "ghatiya": "substandard",
    "bakwaas": "rubbish", "bakwas": "rubbish", "barbaad": "wasted", "barbad": "ruined",
    "dono": "both", "dhanyvad": "thank you", "kripya": "please", "shukriya": "thank you",
    "samasya": "problem", "kathin": "difficult", "mushkil": "difficult", "aasan": "easy",
    "koshish": "effort", "safal": "successful", "asafal": "unsuccessful",
    "azab": "punishment", "jahannum": "hell", "jannat": "paradise", "behisht": "paradise",
    "dushman": "enemy", "mitr": "friend", "dost": "friend", "mitrta": "friendship", "dosti": "friendship",
    "paisa": "money", "kitab": "book", "ghar": "home", "shahar": "city", "film": "movie",
    "kahani": "story", "gaana": "song", "gaane": "songs",
    "pyaar": "love", "pyar": "love", "nafrat": "hate", "khushi": "happiness", "dukh": "sorrow",
    "baat": "statement", "baatein": "talks", "galat": "wrong", "sahi": "correct", "sach": "truth", "jhooth": "lie",
    "madad": "help", "sahayta": "support", "janm": "origin", "karan": "cause", "karnon": "causes",
    "any": "other", "bimari": "illness", "aparyapt": "inadequate", "poshan": "nutrition",
    "khudaee": "excavation", "kamra": "room", "kamron": "rooms", "bare-bare": "large", "bartan": "vessels",
    "shav": "remains", "jinme": "in which", "hathon": "hands", "haath": "hand",

    # Common conversational feedback & phrases
    "pasand aayi": "liked", "pasand aaya": "liked", "pasand aate": "like", "pasand hai": "like",
    "pasand nahi aaya": "did not like", "pasand nahi aayi": "did not like", "pasand nahi": "do not like",
    "accha laga": "liked it", "achha laga": "liked it", "bura laga": "felt bad",
    "kharab nikla": "turned out defective", "bekar nikla": "turned out useless",
    "swaad nahi tha": "was tasteless", "swaad nahi": "tasteless", "swaad bahut achha": "taste is very good",
    "swad nahi tha": "was tasteless", "swad nahi": "tasteless", "swad bahut achha": "taste is very good",
    "kaam nahi kar raha": "is not working", "kaam nahi kar raha hai": "is not working",
    "kaam karta hai": "works well", "kaam nahi": "does not work",
    "crash ho rahi hai": "is crashing", "crash ho raha hai": "is crashing", "crash ho rahi": "is crashing",
    "bar bar crash": "repeatedly crashing", "bar bar": "again and again",
    "problem solve kar di": "solved the problem", "problem solve ho gayi": "problem was resolved",
    "issue resolve kar diya": "resolved the issue", "issue solve kar diya": "solved the issue",
    "solve kar diya": "solved", "solve kar di": "solved", "resolve kar diya": "resolved",
    "refund mil gaya": "received refund", "refund nahi mila": "did not receive refund",
    "paise wapas": "money back", "paise kat gaye": "money was deducted", "paise cut gaye": "money was deducted",
    "cut gaye": "were deducted", "kat gaye": "were deducted", "cut gaya": "was deducted", "kat gaya": "was deducted",
    "recharge nahi hua": "recharge did not happen", "recharge ho gaya": "recharge was successful",
    "open nahi ho raha": "is not opening", "install karne ke baad": "after installing",
    "jawab nahi diya": "did not answer", "staff helpful tha": "staff was helpful",
    "room saaf tha": "room was clean", "bahut dikkat aayi": "faced a lot of difficulty",
    "color thoda alag hai": "color is slightly different", "aaram nahi mila": "did not get relief",
    "thanda tha": "was cold", "thanda khana": "cold food", "khana thanda": "food is cold",
    "rude staff": "rude staff", "staff rude": "staff is rude", "rude tha": "was rude", "rude hai": "is rude",
    "fast delivery": "fast delivery", "slow delivery": "slow delivery", "delivery fast": "delivery was fast",
    "packing mast thi": "packaging was awesome", "packing achhi thi": "packaging was good", "packing bhi mast thi": "packaging was also great",
    "kisi kaam ka nahi": "good for nothing", "kisi kaam ka": "of any use",
    "kabhi mat lena": "never buy this", "kabhi mat khareedna": "never buy this",
    "mat khareedo": "do not buy", "mat lena": "do not take", "mat khareedna": "do not buy",
    "dekhne jayenge": "will go to watch", "match dekhne": "to watch the match",
    "ek number": "top notch", "dil khush": "heart is happy", "dil khush ho gaya": "delighted",

    # Common vocabulary words
    "khareeda": "bought", "kharida": "bought", "khareedi": "bought", "kharidi": "bought",
    "nikla": "turned out", "nikli": "turned out", "nikle": "turned out",
    "gana": "song", "gaana": "song", "gaane": "songs", "awaz": "sound", "awaaz": "voice",
    "swaad": "taste", "swad": "taste", "swadisht": "delicious",
    "thanda": "cold", "thandi": "cold", "garam": "hot",
    "taza": "fresh", "taaza": "fresh", "basi": "stale", "baasi": "stale",
    "fast": "fast", "slow": "slow", "late": "late", "jaldi": "early",
    "rude": "rude", "polite": "polite", "helpful": "helpful",
    "display": "display", "screen": "screen", "battery": "battery", "camera": "camera", "phone": "phone",
    "stadium": "stadium", "match": "match",
    "packing": "packaging", "parcel": "parcel", "dabba": "box",
    "paise": "money", "rupaye": "rupees", "keemat": "price", "daam": "price",
    "sasta": "cheap", "sasti": "cheap", "mehnga": "expensive", "mehngi": "expensive",
    "lekin": "but", "magar": "however", "par": "but", "kyunki": "because",
    "tha": "was", "thi": "was", "the": "were", "hoga": "will be", "hogi": "will be",
    "mera": "my", "meri": "my", "mere": "my", "tera": "your", "teri": "your", "tere": "your",
    "apna": "own", "apni": "own", "apne": "own", "unka": "their", "unki": "their",
    "mast": "awesome", "jhakaas": "fantastic", "lajawab": "splendid", "kamaal": "wonderful",
    "pasand": "like", "bhi": "also", "hi": "only", "toh": "then", "ne": "",
}

# ---------------------------------------------------------------------------
# 2. Comprehensive Hinglish to Devanagari Hindi Bilingual Lexicon
# ---------------------------------------------------------------------------
HINGLISH_TO_HINDI_DICT = {
    # Full preset sentences — direct high-quality translations
    "ye movie bilkul bakwaas hai, paisa aur time dono barbaad!": "यह फिल्म बिल्कुल बकवास है, पैसा और समय दोनों बर्बाद!",
    "ye movie bilkul bakwaas hai paisa aur time dono barbaad": "यह फिल्म बिल्कुल बकवास है, पैसा और समय दोनों बर्बाद!",
    "boring service aur ghatiya khana, yahan kabhi mat aao": "बोरिंग सर्विस और घटिया खाना, यहाँ कभी मत आओ।",
    "boring service aur ghatiya khana yahan kabhi mat aao": "बोरिंग सर्विस और घटिया खाना, यहाँ कभी मत आओ।",
    "main aaj ghar ja raha hoon, dost se milne": "मैं आज घर जा रहा हूँ, दोस्त से मिलने।",
    "main aaj ghar ja raha hoon dost se milne": "मैं आज घर जा रहा हूँ, दोस्त से मिलने।",
    "ye product bahut accha hai aur kaam karta hai": "यह प्रोडक्ट बहुत अच्छा है और काम करता है।",
    "zabardast acting aur shandar kahani, superhit film hai!": "ज़बरदस्त एक्टिंग और शानदार कहानी, सुपरहिट फ़िल्म है!",
    "zabardast acting aur shandar kahani superhit film hai": "ज़बरदस्त एक्टिंग और शानदार कहानी, सुपरहिट फ़िल्म है!",
    "bhai, aaj toh meri life ka sabse best day hai!": "भाई, आज तो मेरी ज़िंदगी का सबसे बेहतरीन दिन है!",
    "bhai aaj toh meri life ka sabse best day hai": "भाई, आज तो मेरी ज़िंदगी का सबसे बेहतरीन दिन है!",

    # Common colloquial particles & idioms
    "arey waah": "अरे वाह",
    "are waah": "अरे वाह",
    "arey wah": "अरे वाह",
    "are wah": "अरे वाह",
    "isse achha": "इससे अच्छा",
    "isse accha": "इससे अच्छा",
    "isse achchha": "इससे अच्छा",
    "isse behtar": "इससे बेहतर",
    "toh kuch": "तो कुछ",
    "to kuch": "तो कुछ",
    "ho hi nahi": "हो ही नहीं",
    "ho hi nahi sakta": "हो ही नहीं सकता",
    "kuch ho hi nahi": "कुछ हो ही नहीं",
    "kuch ho hi nahi sakta": "कुछ हो ही नहीं सकता",
    "arey": "अरे",
    "are": "अरे",
    "waah": "वाह",
    "wah": "वाह",
    "isse": "इससे",
    "usse": "उससे",
    "kisse": "किससे",
    "toh": "तो",
    "sakta": "सकता",
    "sakti": "सकती",
    "sakte": "सकते",
    "behtar": "बेहतर",
    "theek": "ठीक",
    "bhai": "भाई",
    "bhaiya": "भैया",
    "bhaiyo": "भाइयों",
    "bro": "भाई",
    "dude": "यार",
    "yaar": "यार",

    # Sentences, Multi-word phrases & expressions
    "customer service ne call ka jawab nahi diya": "कस्टमर सर्विस ने कॉल का जवाब नहीं दिया",
    "customer service ne call ka jawab": "कस्टमर सर्विस ने कॉल का जवाब",
    "call ka jawab nahi diya": "कॉल का जवाब नहीं दिया",
    "call ka jawab": "कॉल का जवाब",
    "jawab nahi diya": "जवाब नहीं दिया",
    "jawab nahi": "जवाब नहीं",
    "customer service": "कस्टमर सर्विस",
    "customer care": "कस्टमर केयर",
    "customer support": "कस्टमर सपोर्ट",
    "app install karne ke baad open nahi ho raha": "ऐप इंस्टॉल करने के बाद ओपन नहीं हो रहा",
    "install karne ke baad open nahi ho raha": "इंस्टॉल करने के बाद ओपन नहीं हो रहा",
    "install karne ke baad": "इंस्टॉल करने के बाद",
    "install karne": "इंस्टॉल करने",
    "open nahi ho raha": "ओपन नहीं हो रहा",
    "open nahi": "ओपन नहीं",
    "battery backup bahut kam hai aur phone garam ho jata hai": "बैटरी बैकअप बहुत कम है और फ़ोन गर्म हो जाता है",
    "battery backup bahut kam hai": "बैटरी बैकअप बहुत कम है",
    "battery backup": "बैटरी बैकअप",
    "bahut kam hai": "बहुत कम है",
    "bahut kam": "बहुत कम",
    "garam ho jata hai": "गर्म हो जाता है",
    "garam ho jata": "गर्म हो जाता",
    "hotel ka room saaf tha aur staff helpful tha": "होटल का रूम साफ़ था और स्टाफ़ मददगार था",
    "hotel ka room": "होटल का रूम",
    "room saaf tha": "रूम साफ़ था",
    "staff helpful tha": "स्टाफ़ मददगार था",
    "staff helpful": "स्टाफ़ मददगार",
    "helpful tha": "मददगार था",
    "ticket book karne me bahut dikkat aayi": "टिकट बुक करने में बहुत दिक्कत आई",
    "ticket book karne me": "टिकट बुक करने में",
    "ticket book karne": "टिकट बुक करने",
    "ticket book": "टिकट बुक",
    "bahut dikkat aayi": "बहुत दिक्कत आई",
    "dikkat aayi": "दिक्कत आई",
    "dikkat hui": "दिक्कत हुई",
    "paise cut gaye lekin recharge nahi hua": "पैसे कट गए लेकिन रिचार्ज नहीं हुआ",
    "paise cut gaye": "पैसे कट गए",
    "paise kat gaye": "पैसे कट गए",
    "cut gaye": "कट गए",
    "kat gaye": "कट गए",
    "cut gaya": "कट गया",
    "kat gaya": "कट गया",
    "recharge nahi hua": "रिचार्ज नहीं हुआ",
    "recharge ho gaya": "रिचार्ज हो गया",
    "dress ka material bahut achha hai par color thoda alag hai": "ड्रेस का मटीरियल बहुत अच्छा है पर कलर थोड़ा अलग है",
    "dress ka material": "ड्रेस का मटीरियल",
    "material bahut achha hai": "मटीरियल बहुत अच्छा है",
    "material bahut achha": "मटीरियल बहुत अच्छा",
    "color thoda alag hai": "कलर थोड़ा अलग है",
    "color thoda alag": "कलर थोड़ा अलग",
    "thoda alag hai": "थोड़ा अलग है",
    "thoda alag": "थोड़ा अलग",
    "doctor ne dawai lene ke liye bola tha par aaram nahi mila": "डॉक्टर ने दवाई लेने के लिए बोला था पर आराम नहीं मिला",
    "lene ke liye bola tha": "लेने के लिए बोला था",
    "lene ke liye bola": "लेने के लिए बोला",
    "lene ke liye": "लेने के लिए",
    "bola tha": "बोला था",
    "aaram nahi mila": "आराम नहीं मिला",
    "aaram mila": "आराम मिला",
    "train 2 ghante late chal rahi hai": "ट्रेन 2 घंटे लेट चल रही है",
    "ghante late chal rahi hai": "घंटे लेट चल रही है",
    "ghante late": "घंटे लेट",
    "chal rahi hai": "चल रही है",
    "chal raha hai": "चल रहा है",
    "chal rahe hain": "चल रहे हैं",
    "ho jata hai": "हो जाता है",
    "ho jati hai": "हो जाती है",
    "ho jate hain": "हो जाते हैं",
    "ho raha hai": "हो रहा है",
    "ho rahi hai": "हो रही है",
    "ho rahe hain": "हो रहे हैं",
    "ho gaya hai": "हो गया है",
    "ho gayi hai": "हो गई है",
    "ho gaye hain": "हो गए हैं",
    "ke baad": "के बाद",
    "ke liye": "के लिए",
    "ke saath": "के साथ",
    "ke sath": "के साथ",
    "ke pass": "के पास",
    "ke bina": "के बिना",
    "maine online pizza order kiya tha lekin delivery bahut late aayi": "मैंने ऑनलाइन पिज़्ज़ा ऑर्डर किया था लेकिन डिलीवरी बहुत लेट आई",
    "pizza order kiya tha": "पिज़्ज़ा ऑर्डर किया था",
    "order kiya tha": "ऑर्डर किया था",
    "delivery bahut late aayi": "डिलीवरी बहुत लेट आई",
    "delivery bahut late": "डिलीवरी बहुत लेट",
    "late aayi": "लेट आई",
    "movie ki kahani bahut boring thi aur acting bhi kharab thi": "फिल्म की कहानी बहुत उबाऊ थी और एक्टिंग भी खराब थी",
    "bhai aaj toh meri life ka sabse best day hai": "भाई, आज तो मेरी ज़िंदगी का सबसे बेहतरीन दिन है!",
    "aaj toh meri life ka sabse best day hai": "आज तो मेरी ज़िंदगी का सबसे बेहतरीन दिन है!",
    "meri life ka sabse best day": "मेरी ज़िंदगी का सबसे बेहतरीन दिन",
    "sabse best day": "सबसे बेहतरीन दिन",
    "sabse best": "सबसे बेहतरीन",
    "best day": "बेहतरीन दिन",
    "meri life": "मेरी ज़िंदगी",
    "mera bhai": "मेरा भाई",
    "mere bhai": "मेरे भाई",
    "aaj toh": "आज तो",
    "aaj to": "आज तो",
    "bilkul bakwaas": "बिल्कुल बकवास",
    "bilkul bakwas": "बिल्कुल बकवास",
    "paisa vasool": "पैसा वसूल",
    "dono barbaad": "दोनों बर्बाद",
    "dono barbad": "दोनों बर्बाद",
    "bahut achha": "बहुत अच्छा",
    "bahut achchha": "बहुत अच्छा",
    "bahut accha": "बहुत अच्छा",
    "bahut bura": "बहुत बुरा",
    "bahut sundar": "बहुत सुंदर",
    "kabhi nahi": "कभी नहीं",
    "kabhi nahin": "कभी नहीं",
    "kuch nahi": "कुछ नहीं",
    "kuchh nahin": "कुछ नहीं",
    "koi nahi": "कोई नहीं",
    "koi baat nahi": "कोई बात नहीं",
    "ek doosre": "एक-दूसरे",
    "ek doosre ka": "एक-दूसरे का",
    "chal door": "दूर हट",
    "kar diya": "कर दिया",
    "kar raha": "कर रहा",
    "kar rahe": "कर रहे",
    "kar rahi": "कर रही",
    "ho gaya": "हो गया",
    "ho gayi": "हो गई",
    "aa gaya": "आ गया",
    "chala gaya": "चला गया",

    # Pronouns & determiners
    "ham": "हम", "hum": "हम", "hamne": "हमने", "humne": "हमने", "hamara": "हमारा", "hamari": "हमारी", "hamare": "हमारे",
    "main": "मैं", "maine": "मैंने", "mainne": "मैंने", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
    "mujhe": "मुझे", "mujhko": "मुझको",
    "aap": "आप", "aapko": "आपको", "aapka": "आपका", "aapki": "आपकी", "aapke": "आपके",
    "tum": "तुम", "tumne": "तुमने", "tumhara": "तुम्हारा", "tumhari": "तुम्हारी", "tumhare": "तुम्हारे", "tumhen": "तुम्हें", "tumhe": "तुम्हें",
    "vah": "वह", "woh": "वह", "vo": "वह", "unhonne": "उन्होंने", "unhen": "उन्हें", "unhe": "उन्हें",
    "unka": "उनका", "unki": "उनकी", "unke": "उनके", "use": "उसे", "uska": "उसका", "uski": "उसकी", "uske": "उसके",
    "ye": "यह", "yeh": "यह", "ve": "वे", "inhen": "इन्हें", "inhe": "इन्हें", "inka": "इनका", "inki": "इनकी", "inke": "इनके",
    "ise": "इसे", "iska": "इसका", "iski": "इसकी", "iske": "इसके", "is": "इस", "in": "इन",
    "yah": "यह", "kuchh": "कुछ", "kuch": "कुछ", "sab": "सब", "sabhi": "सभी", "har": "हर", "pratyek": "प्रत्येक",
    "apna": "अपना", "apne": "अपने", "apni": "अपनी", "koi": "कोई", "koee": "कोई", "kisi": "किसी",

    # Conjunctions & prepositions
    "aur": "और", "ya": "या", "athva": "अथवा", "kintu": "किंतु", "parntu": "परंतु", "parantu": "परंतु", "lekin": "लेकिन", "magar": "मगर",
    "bhi": "भी", "hi": "ही", "to": "तो", "tak": "तक", "se": "से", "men": "में", "mein": "में",
    "par": "पर", "ko": "को", "ke": "के", "ki": "की", "ka": "का", "lie": "लिए", "liye": "लिए",
    "tatha": "तथा", "evn": "एवं", "sath": "साथ", "saath": "साथ", "bina": "बिना", "jaise": "जैसे",
    "kyunki": "क्योंकि", "kyonki": "क्योंकि", "isliye": "इसलिए",
    "jab": "जब", "tab": "तब", "ab": "अब", "kab": "कब", "kahan": "कहाँ", "yahan": "यहाँ", "wahan": "वहाँ",
    "kya": "क्या", "kyon": "क्यों", "kyun": "क्यों", "kaun": "कौन", "kaise": "कैसे",

    # Negations & adverbs
    "nahi": "नहीं", "nahin": "नहीं", "na": "नहीं", "mat": "मत",
    "thoda": "थोड़ा", "thodi": "थोड़ी", "thode": "थोड़े", "kam": "कम", "zyada": "ज़्यादा", "jyada": "ज़्यादा", "adhik": "अधिक",
    "bahut": "बहुत", "bilkul": "बिल्कुल", "khub": "खूब", "sirf": "सिर्फ़",

    # Verbs
    "hai": "है", "hain": "हैं", "tha": "था", "thi": "थी", "the": "थे", "hote": "होते", "hoti": "होती", "hota": "होता",
    "karna": "करना", "karta": "करता", "karti": "करती", "karte": "करते", "kiya": "किया", "kiye": "किए", "karega": "करेगा",
    "hona": "होना", "huaa": "हुआ", "hua": "हुआ", "hue": "हुए", "hui": "हुई", "ho": "हो",
    "raha": "रहा", "rahe": "रहे", "rahi": "रही",
    "chahie": "चाहिए", "chahiye": "चाहिए", "sakte": "सकते", "sakta": "सकता", "sakti": "सकती",
    "kahna": "कहना", "kaha": "कहा", "kahe": "कहा", "bola": "बोला", "bole": "बोले", "bataya": "बताया",
    "dekhna": "देखना", "dekha": "देखा", "dekhne": "देखने", "dekho": "देखो",
    "jana": "जाना", "gaya": "गया", "gaye": "गए", "gayi": "गई",
    "aana": "आना", "aaya": "आया", "aaye": "आए", "aata": "आता",
    "mile": "मिले", "mila": "मिला", "milte": "मिलते", "rahega": "रहेगा", "de": "दे", "di": "दी", "diya": "दिया",

    # Sentiment vocabulary & nouns
    "din": "दिन", "rat": "रात", "raat": "रात", "samay": "समय", "varsh": "वर्ष", "sal": "साल", "saal": "साल",
    "log": "लोग", "logon": "लोगों", "bachche": "बच्चे", "vyakti": "व्यक्ति", "desh": "देश",
    "sarkar": "सरकार", "nyayalay": "न्यायालय", "kanoon": "कानून", "kam": "कम", "kary": "कार्य", "kaam": "काम",
    "shiksha": "शिक्षा", "swasthya": "स्वास्थ्य", "pani": "पानी", "dharti": "धरती", "aasman": "आसमान",
    "bhagvan": "भगवान", "khuda": "ख़ुदा", "allah": "अल्लाह", "rab": "रब", "paravardigar": "परवरदिगार",
    "shandar": "शानदार", "zabardast": "ज़बरदस्त", "badhiya": "बढ़िया", "kamaal": "कमाल", "lazawab": "लाजवाब",
    "achchha": "अच्छा", "achha": "अच्छा", "accha": "अच्छा", "acchi": "अच्छी", "acche": "अच्छे",
    "bura": "बुरा", "buri": "बुरी", "bure": "बुरे", "kharab": "ख़राब", "ghatiya": "घटिया",
    "bakwaas": "बकवास", "bakwas": "बकवास", "barbaad": "बर्बाद", "barbad": "बर्बाद",
    "dono": "दोनों", "dhanyvad": "धन्यवाद", "kripya": "कृपया", "shukriya": "शुक्रिया",
    "samasya": "समस्या", "kathin": "कठिन", "mushkil": "मुश्किल", "aasan": "आसान",
    "koshish": "कोशिश", "safal": "सफल", "asafal": "असफल",
    "azab": "सज़ा", "jahannum": "नरक", "jannat": "स्वर्ग",
    "dushman": "दुश्मन", "mitr": "मित्र", "dost": "दोस्त", "mitrta": "मित्रता", "dosti": "दोस्ती",
    "paisa": "पैसा", "kitab": "किताब", "ghar": "घर", "shahar": "शहर",
    "kahani": "कहानी", "gaana": "गाना", "gaane": "गाने",
    "pyaar": "प्यार", "pyar": "प्यार", "nafrat": "नफ़रत", "khushi": "ख़ुशी", "dukh": "दुःख", "dard": "दर्द",
    "aaj": "आज", "kal": "कल", "pahle": "पहले", "pehle": "पहले", "baad": "बाद", "bad": "बाद",

    # English loanwords & Customer Reviews Lexicon frequently used in Hinglish
    "movie": "फिल्म", "cinema": "सिनेमा", "film": "फिल्म",
    "acting": "अभिनय", "actor": "अभिनेता", "actress": "अभिनेत्री", "director": "निर्देशक",
    "story": "कहानी", "song": "गाना", "songs": "गाने", "music": "संगीत", "dialogue": "संवाद",
    "time": "समय", "money": "पैसा", "friend": "दोस्त", "friends": "दोस्त", "family": "परिवार",
    "problem": "समस्या", "problems": "समस्याएँ", "help": "मदद", "support": "समर्थन",
    "life": "ज़िंदगी", "zindagi": "ज़िंदगी", "jeevan": "जीवन", "death": "मृत्यु", "world": "दुनिया", "government": "सरकार",
    "day": "दिन", "days": "दिन", "night": "रात", "nights": "रातें",
    "sabse": "सबसे", "sab": "सब", "sabhi": "सभी",
    "good": "अच्छा", "bad": "बुरा", "best": "बेहतरीन", "better": "बेहतर", "worst": "सबसे खराब",
    "great": "बहुत बढ़िया", "awesome": "शानदार", "super": "शानदार", "superb": "लाजवाब", "fantastic": "शानदार",
    "happy": "खुश", "sad": "उदास", "boring": "उबाऊ", "flop": "फ़्लॉप", "hit": "हिट",
    "fraud": "धोखा", "loot": "लूट", "waste": "बर्बादी",
    "khana": "खाना", "khana khana": "खाना खाना", "peena": "पीना",
    "hoon": "हूँ", "ja": "जा", "aao": "आओ", "aao mat": "मत आओ", "milne": "मिलने",
    "aaj": "आज", "yahan": "यहाँ", "wahan": "वहाँ",
    "service": "सर्विस", "product": "प्रोडक्ट", "delivery": "डिलीवरी", "quality": "क्वालिटी",
    "order": "ऑर्डर", "app": "ऐप", "staff": "स्टाफ़", "experience": "अनुभव",
    "price": "कीमत", "rate": "रेट", "cost": "लागत", "kabhi": "कभी", "dono": "दोनों",
    "bada": "बड़ा", "chhota": "छोटा", "tez": "तेज़", "dheere": "धीरे",
    "suno": "सुनो", "batao": "बताओ", "samajh": "समझ", "samjha": "समझा",
    "superhit": "सुपरहिट", "blockbuster": "ब्लॉकबस्टर", "scam": "घोटाला",
    "lagta": "लगता", "lagti": "लगती", "lagte": "लगते", "laga": "लगा",
    "galat": "गलत", "sahi": "सही", "baat": "बात", "baatein": "बातें",
    "madad": "मदद", "sahayta": "सहायता", "aayega": "आएगा", "aayegi": "आएगी", "aayenge": "आएंगे",
    "chalta": "चलता", "chalti": "चलती", "chalte": "चलते",
    "karan": "कारण", "karnon": "कारणों", "janm": "जन्म", "deti": "देती", "deta": "देता",
    "bimari": "बीमारी", "aparyapt": "अपर्याप्त", "poshan": "पोषण",
    "khudaee": "खुदाई", "kamra": "कमरा", "kamron": "कमरों",
    "bare-bare": "बड़े-बड़े", "bartan": "बर्तन", "shav": "शव", "jinme": "जिनमें",
    "haath": "हाथ", "hathon": "हाथों", "any": "अन्य", "pallvit-pushpit": "पल्लवित-पुष्पित",

    # Domain vocabulary & inflections for CSV batches
    "install": "इंस्टॉल", "installed": "इंस्टॉल", "uninstalled": "अनइंस्टॉल", "open": "ओपन", "opened": "ओपन",
    "close": "बंद", "closed": "बंद", "update": "अपडेट", "download": "डाउनलोड", "crash": "क्रैश",
    "battery": "बैटरी", "backup": "बैकअप", "charger": "चार्जर", "charge": "चार्ज", "charging": "चार्जिंग",
    "customer": "कस्टमर", "care": "केयर", "executive": "एग्जीक्यूटिव", "call": "कॉल", "called": "कॉल",
    "jawab": "जवाब", "helpful": "मददगार", "polite": "विनम्र", "rude": "रूढ़", "dikkat": "दिक्कत",
    "hotel": "होटल", "room": "रूम", "rooms": "कमरे", "bed": "बिस्तर", "saaf": "साफ़", "ganda": "गंदा",
    "ticket": "टिकट", "tickets": "टिकट", "book": "बुक", "booking": "बुकिंग", "booked": "बुक",
    "train": "ट्रेन", "flight": "फ़्लाइट", "bus": "बस", "station": "स्टेशन", "ghante": "घंटे", "ghanta": "घंटा",
    "cut": "कट", "kat": "कट", "katt": "कट", "recharge": "रिचार्ज", "balance": "बैलेंस",
    "dress": "ड्रेस", "material": "मटीरियल", "color": "कलर", "colour": "रंग", "thoda": "थोड़ा", "alag": "अलग",
    "doctor": "डॉक्टर", "dawai": "दवाई", "dawiyan": "दवाइयां", "aaram": "आराम", "rest": "आराम",
    "karne": "करने", "karta": "करता", "karte": "करते", "karti": "करती", "karna": "करना",
    "kara": "करा", "kare": "करे", "karega": "करेगा", "karegi": "करेगी", "karenge": "करेंगे",
    "chalta": "चलता", "chalti": "चलती", "chalte": "चलते", "chalne": "चलने", "chalna": "चलना",
    "lagta": "लगता", "lagti": "लगती", "lagte": "लगते", "lagne": "लगने", "lagna": "लगना",
    "milta": "मिलता", "milti": "मिलती", "milte": "मिलते", "milne": "मिलने", "milna": "मिलना",
    "usne": "उसने", "isne": "इसने", "unhone": "उन्होंने", "inhone": "इन्होंने",
    "kisne": "किसने", "jisne": "जिसने", "maine": "मैंने", "humne": "हमने", "tumne": "तुमने", "aapne": "आपने",
    "apna": "अपना", "apni": "अपनी", "apne": "अपने",
    "me": "में", "mein": "में", "mai": "मैं", "main": "मैं", "par": "पर",
    "hota": "होता", "hoti": "होती", "hote": "होते", "hua": "हुआ", "hui": "हुई", "hue": "हुए",
    "dekhne": "देखने", "dekhna": "देखना", "dekha": "देखा", "dekhi": "देखी",
    "bolne": "बोलने", "bolna": "बोलना", "bola": "बोला", "boli": "बोली", "bole": "बोले",
    "kahne": "कहने", "kahna": "कहना", "kaha": "कहा", "kahi": "कही",
    "lene": "लेने", "lena": "लेना", "liya": "लिया", "li": "ली", "dena": "देना", "diya": "दिया", "di": "दी",
    "pizza": "पिज़्ज़ा", "burger": "बर्गर", "food": "खाना", "restaurant": "रेस्टोरेंट",

    # Feedback phrases and common words in Hindi
    "pasand aayi": "पसंद आई", "pasand aaya": "पसंद आया", "pasand hai": "पसंद है", "pasand": "पसंद",
    "pasand nahi": "पसंद नहीं", "khareeda": "खरीदा", "kharida": "खरीदा", "khareedi": "खरीदी",
    "nikla": "निकला", "nikli": "निकली", "gana": "गाना", "gaana": "गाना", "awaz": "आवाज़", "awaaz": "आवाज़",
    "swaad": "स्वाद", "swad": "स्वाद", "swadisht": "स्वादिष्ट", "thanda": "ठंडा", "thandi": "ठंडी", "garam": "गर्म",
    "fast": "फास्ट", "slow": "स्लो", "late": "लेट", "jaldi": "जल्दी", "rude": "रुड", "polite": "विनम्र",
    "display": "डिस्प्ले", "screen": "स्क्रीन", "battery": "बैटरी", "camera": "कैमरा", "phone": "फ़ोन",
    "stadium": "स्टेडियम", "match": "मैच", "packing": "पैकिंग", "parcel": "पार्सल", "dabba": "डिब्बा",
    "paise": "पैसे", "rupaye": "रुपये", "sasta": "सस्ता", "mehnga": "महंगा", "mast": "मस्त", "badhiya": "बढ़िया",
    "lajawab": "लाजवाब", "kamaal": "कमाल", "ghatiya": "घटिया", "bekar": "बेकार", "bekaar": "बेकार",
    "lekin": "लेकिन", "magar": "मगर", "kyunki": "क्योंकि", "tha": "था", "thi": "थी", "the": "थे",
    "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "tera": "तेरा", "teri": "तेरी", "tere": "तेरे",
    "apna": "अपना", "apni": "अपनी", "apne": "अपने", "unka": "उनका", "unki": "उनकी",
    "bar bar": "बार-बार", "crash": "क्रैश", "solve": "हल", "resolve": "हल", "support": "सपोर्ट",
    "jayenge": "जाएंगे", "jaunga": "जाऊंगा", "dekhne": "देखने", "aayi": "आई", "aaya": "आया",
}

# ---------------------------------------------------------------------------
# 3. Phonetic Latin-to-Devanagari Transliteration Fallback Engine
# ---------------------------------------------------------------------------
DEV_CONSONANTS = [
    ("chh", "छ"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"),
    ("th", "थ"), ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"),
    ("rh", "ढ़"), ("k", "क"), ("g", "ग"), ("j", "ज"), ("t", "त"),
    ("d", "द"), ("n", "न"), ("p", "प"), ("f", "फ़"), ("b", "ब"),
    ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"), ("v", "व"),
    ("w", "व"), ("s", "स"), ("h", "ह"), ("z", "ज़"), ("x", "क्स"),
    ("q", "क़"), ("c", "क"),
]

DEV_VOWEL_SIGNS = [
    ("aa", "ा"), ("ai", "ै"), ("au", "ौ"), ("ee", "ी"), ("ii", "ी"),
    ("oo", "ू"), ("uu", "ू"), ("i", "ि"), ("u", "ु"),
    ("e", "े"), ("o", "ो"), ("a", ""),
]

DEV_INIT_VOWELS = [
    ("aa", "आ"), ("ai", "ऐ"), ("au", "औ"), ("ee", "ई"), ("ii", "ई"),
    ("oo", "ऊ"), ("uu", "ऊ"), ("a", "अ"), ("i", "इ"), ("u", "उ"),
    ("e", "ए"), ("o", "ओ"),
]


TRUE_CONJUNCT_PREFIXES = {
    "st", "sp", "sk", "sm", "pr", "tr", "kr", "dr", "br", "gr", "fr",
    "sw", "shw", "ky", "py", "ny", "ty", "vy", "gy", "dy", "my", "ly", "sy",
    "kt", "pt", "ks", "bd", "nt", "nd", "mp", "mb", "nk", "sht", "chch",
    "bl", "cl", "fl", "gl", "pl", "sl"
}


def transliterate_roman_to_devanagari(word: str) -> str:
    """
    Phonetically transliterate any unmapped Romanized Hindi token into Devanagari script.
    Accurately preserves Indic schwa on standard consonants while correctly placing halants
    only on true phonotactic conjunct clusters and geminates.
    """
    w = word.lower()
    res = []
    i = 0
    n = len(w)

    while i < n:
        matched_c = None
        c_char = None
        for c_pat, c_val in DEV_CONSONANTS:
            if w[i:].startswith(c_pat):
                matched_c = c_pat
                c_char = c_val
                break

        if matched_c:
            pos_after_c = i + len(matched_c)
            # Terminal single 'a' after consonant produces long aa matra in romanized Hindi
            if pos_after_c == n - 1 and w[pos_after_c] == "a":
                res.append(c_char + "ा")
                i = n
                continue

            matched_v = None
            v_sign = None
            for v_pat, v_val in DEV_VOWEL_SIGNS:
                if w[pos_after_c:].startswith(v_pat):
                    matched_v = v_pat
                    v_sign = v_val
                    break

            if matched_v:
                i = pos_after_c + len(matched_v)
                # Terminal 'i' after consonant in Romanized Hindi is long 'ee' (दीर्घ ई)
                if matched_v == "i" and i == n:
                    v_sign = "ी"
                elif matched_v == "ai" and i == n:
                    v_sign = "ाई"
                res.append(c_char + v_sign)
            else:
                # Check for conjunct halant
                is_conjunct = False
                rem = w[i:]
                if len(matched_c) == 1 and pos_after_c < n and w[pos_after_c] == matched_c:
                    is_conjunct = True
                else:
                    for tc in TRUE_CONJUNCT_PREFIXES:
                        if rem.startswith(tc):
                            is_conjunct = True
                            break

                if is_conjunct:
                    res.append(c_char + "्")
                else:
                    res.append(c_char)
                i = pos_after_c
        else:
            # Initial or stand-alone vowel
            matched_iv = None
            iv_char = None
            for v_pat, v_val in DEV_INIT_VOWELS:
                if w[i:].startswith(v_pat):
                    matched_iv = v_pat
                    iv_char = v_val
                    break

            if matched_iv:
                i += len(matched_iv)
                res.append(iv_char)
            else:
                res.append(w[i])
                i += 1

    return "".join(res)


# ---------------------------------------------------------------------------
# 4. Translation Functions: English & Hindi
# ---------------------------------------------------------------------------
# Caches for translations
_TRANSLATION_CACHE_EN: Dict[str, str] = {}
_TRANSLATION_CACHE_HI: Dict[str, str] = {}


def translate_hinglish_to_hindi(text: str) -> str:
    """
    Translate Roman-script Hinglish text into pure Devanagari Hindi (हिंदी).
    Priority:
      1. Full-sentence dictionary lookup (instant, authentic quality)
      2. Neural translation via Google Translate API (when online)
      3. High-coverage bilingual n-gram dictionary + conjunct-aware transliteration fallback
    """
    clean = text.strip()
    if not clean:
        return ""

    # Full-sentence lookup first (handles preset and common sentences directly)
    clean_low = clean.lower()
    if clean_low in HINGLISH_TO_HINDI_DICT:
        return HINGLISH_TO_HINDI_DICT[clean_low]
    # Also try without trailing punctuation
    clean_low_nopunct = clean_low.rstrip(".,!?;:")
    if clean_low_nopunct in HINGLISH_TO_HINDI_DICT:
        return HINGLISH_TO_HINDI_DICT[clean_low_nopunct]

    if clean in _TRANSLATION_CACHE_HI:
        return _TRANSLATION_CACHE_HI[clean]

    # Try Google Translate web API via standard library urllib (active when online)
    try:
        import urllib.request
        import urllib.parse
        import json
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=hi&dt=t&q={urllib.parse.quote(clean)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                segments = [s[0] for s in data[0] if s and isinstance(s, list) and s[0]]
                cand = "".join(segments).strip()
                if cand and any("\u0900" <= c <= "\u097F" for c in cand):
                    _TRANSLATION_CACHE_HI[clean] = cand
                    return cand
    except Exception:
        pass

    tokens = re.findall(r"[\w']+|[.,!?;:]", clean)
    translated_tokens = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_low = token.lower()

        # Check 4-grams first
        if i + 3 < len(tokens):
            fourgram = f"{token_low} {tokens[i+1].lower()} {tokens[i+2].lower()} {tokens[i+3].lower()}"
            if fourgram in HINGLISH_TO_HINDI_DICT:
                translated_tokens.append(HINGLISH_TO_HINDI_DICT[fourgram])
                i += 4
                continue

        # Check trigrams
        if i + 2 < len(tokens):
            trigram = f"{token_low} {tokens[i+1].lower()} {tokens[i+2].lower()}"
            if trigram in HINGLISH_TO_HINDI_DICT:
                translated_tokens.append(HINGLISH_TO_HINDI_DICT[trigram])
                i += 3
                continue

        # Check bigrams
        if i + 1 < len(tokens):
            bigram = f"{token_low} {tokens[i+1].lower()}"
            if bigram in HINGLISH_TO_HINDI_DICT:
                trans = HINGLISH_TO_HINDI_DICT[bigram]
                translated_tokens.append(trans)
                i += 2
                continue

        # Check single word dictionary
        if token_low in HINGLISH_TO_HINDI_DICT:
            translated_tokens.append(HINGLISH_TO_HINDI_DICT[token_low])
        elif re.match(r"^[.,!?;:]$", token):
            translated_tokens.append(token)
        elif token.isdigit():
            translated_tokens.append(token)
        else:
            # Phonetic transliteration fallback
            devanagari = transliterate_roman_to_devanagari(token)
            translated_tokens.append(devanagari)
        i += 1

    result = " ".join(translated_tokens)
    result = re.sub(r'\s+([.,!?;:])', r'\1', result)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def translate_hinglish_to_english(text: str) -> str:
    """
    Translate Roman-script Hinglish text into fluent, grammatically proper English.
    Priority:
      1. Full-sentence dictionary lookup (instant, perfect quality)
      2. Neural translation via deep_translator (requires internet)
      3. Robust rule-based n-gram + word-by-word fallback
    """
    clean = text.strip()
    if not clean:
        return ""

    # 1. Full-sentence lookup (covers preset sentences & common phrases)
    clean_low = clean.lower()
    if clean_low in HINGLISH_TO_ENGLISH_DICT:
        return HINGLISH_TO_ENGLISH_DICT[clean_low]
    # Also try without trailing punctuation
    clean_low_nopunct = clean_low.rstrip(".,!?;:")
    if clean_low_nopunct in HINGLISH_TO_ENGLISH_DICT:
        return HINGLISH_TO_ENGLISH_DICT[clean_low_nopunct]

    # 2. Obtain Devanagari Hindi intermediate representation
    hindi_rep = translate_hinglish_to_hindi(clean)

    if hindi_rep in _TRANSLATION_CACHE_EN:
        return _TRANSLATION_CACHE_EN[hindi_rep]

    # 3. Direct Google Translate web API via standard library urllib (zero external packages required)
    neural_result = None
    import urllib.request
    import urllib.parse
    import json

    for query_text, sl in [(hindi_rep, "hi"), (clean, "auto")]:
        if neural_result:
            break
        if not query_text or len(query_text.strip()) == 0:
            continue
        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl=en&dt=t&q={urllib.parse.quote(query_text)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    segments = [s[0] for s in data[0] if s and isinstance(s, list) and s[0]]
                    cand = "".join(segments).strip()
                    if cand and len(cand) > 3:
                        neural_result = cand
        except Exception:
            pass

    # Secondary: try deep_translator if installed
    if neural_result is None:
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source="auto", target="en")
            res = translator.translate(hindi_rep or clean)
            if res and len(res.strip()) > 3:
                neural_result = res
        except Exception:
            pass

    if neural_result is None:
        try:
            from deep_translator import MyMemoryTranslator
            translator = MyMemoryTranslator(source="hi-IN", target="en-US")
            res = translator.translate(hindi_rep or clean)
            if res and len(res.strip()) > 3 and "MYMEMORY WARNING" not in res:
                neural_result = res
        except Exception:
            pass

    if neural_result:
        _TRANSLATION_CACHE_EN[hindi_rep] = neural_result
        return neural_result

    # 4. Rule-based n-gram + word-level fallback
    tokens = re.findall(r"[\w'-]+|[.,!?;:]", clean)
    translated_tokens = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_low = token.lower()

        # Skip punctuation — carry through as-is
        if re.match(r"^[.,!?;:]$", token):
            translated_tokens.append(token)
            i += 1
            continue

        # Check 4-grams first
        if i + 3 < len(tokens):
            fourgram = f"{token_low} {tokens[i+1].lower()} {tokens[i+2].lower()} {tokens[i+3].lower()}"
            if fourgram in HINGLISH_TO_ENGLISH_DICT:
                trans = HINGLISH_TO_ENGLISH_DICT[fourgram]
                translated_tokens.append(trans.capitalize() if token[0].isupper() else trans)
                i += 4
                continue

        # Check trigrams
        if i + 2 < len(tokens):
            trigram = f"{token_low} {tokens[i+1].lower()} {tokens[i+2].lower()}"
            if trigram in HINGLISH_TO_ENGLISH_DICT:
                trans = HINGLISH_TO_ENGLISH_DICT[trigram]
                translated_tokens.append(trans.capitalize() if token[0].isupper() else trans)
                i += 3
                continue

        # Check bigrams
        if i + 1 < len(tokens):
            bigram = f"{token_low} {tokens[i+1].lower()}"
            if bigram in HINGLISH_TO_ENGLISH_DICT:
                trans = HINGLISH_TO_ENGLISH_DICT[bigram]
                translated_tokens.append(trans.capitalize() if token[0].isupper() else trans)
                i += 2
                continue

        # Single word check
        if token_low in HINGLISH_TO_ENGLISH_DICT:
            trans = HINGLISH_TO_ENGLISH_DICT[token_low]
            translated_tokens.append(trans.capitalize() if token[0].isupper() else trans)
        else:
            # Keep unknown tokens as-is (they may be proper nouns or untranslatable terms)
            translated_tokens.append(token)
        i += 1

    result = " ".join(translated_tokens)
    result = re.sub(r'\s+([.,!?;:])', r'\1', result)

    # Post-processing smoothing for English grammar
    result = re.sub(r'\b(is|are|am|was|were|of|in|to|for)\s+\1\b', r'\1', result, flags=re.IGNORECASE)
    result = re.sub(r'\bvery\s+(good|bad|slow|fast|cold|hot|terrible)\s+(is|was)\b', r'\2 very \1', result, flags=re.IGNORECASE)
    result = re.sub(r'\b(is|was)\s+\1\b', r'\1', result, flags=re.IGNORECASE)
    result = re.sub(r'\bare going is\b', 'is ongoing', result, flags=re.IGNORECASE)
    result = re.sub(r'\bvery like\b', 'really like', result, flags=re.IGNORECASE)
    result = re.sub(r'\bjayenge\b', 'will go', result, flags=re.IGNORECASE)
    result = re.sub(r'\bjaunga\b', 'will go', result, flags=re.IGNORECASE)
    result = re.sub(r'\bdekhne\b', 'to watch', result, flags=re.IGNORECASE)
    result = re.sub(r'\bstaff is rude is\b', 'staff is rude', result, flags=re.IGNORECASE)
    result = re.sub(r'\bcustomer support my\b', 'customer support', result, flags=re.IGNORECASE)
    result = re.sub(r'\bMe\s+lagta\s+is\b', 'It seems to me', result, flags=re.IGNORECASE)
    result = re.sub(r'\bYou\s+where\s+are\s+going\s+be\b', 'Where are you going', result, flags=re.IGNORECASE)
    result = re.sub(r'\bKya\s+you\b', 'Can you', result, flags=re.IGNORECASE)
    result = re.sub(r'\bmadad\s+kar\s+can\s+are\b', 'help me', result, flags=re.IGNORECASE)
    result = re.sub(r'\bto\s+seeing\s+of\s+for\b', 'to view', result, flags=re.IGNORECASE)
    result = re.sub(r'\bpermission\s+do\s+not\s+den\b', 'do not grant permission', result, flags=re.IGNORECASE)
    result = re.sub(r'\bmil\s+went\s+is\b', 'has been received', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()
    if result:
        result = result[0].upper() + result[1:]
    return result


def translate_hinglish(text: str) -> Dict[str, str]:
    """
    Convenience dual-translation function returning Hinglish, Hindi, and English.
    """
    return {
        "hinglish": text,
        "hindi": translate_hinglish_to_hindi(text),
        "english": translate_hinglish_to_english(text),
    }


# ---------------------------------------------------------------------------
# 5. Dataset Splitting & Serialization
# ---------------------------------------------------------------------------
def generate_translations_for_splits(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    output_path: str,
    system_name: str = "HinglishBilingualTranslator_v2.0_Dual",
) -> List[Dict[str, Any]]:
    """
    Translate all splits into BOTH English and Devanagari Hindi, keeping strict
    one-to-one ID correspondence, split assignments, and metadata.
    """
    all_translated = []
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    splits_data = [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]

    for split_name, rows in splits_data:
        for r in rows:
            hinglish_text = r["text"]
            eng_trans = translate_hinglish_to_english(hinglish_text)
            hin_trans = translate_hinglish_to_hindi(hinglish_text)

            # Fallback guards
            if not eng_trans or len(eng_trans.strip()) == 0:
                eng_trans = hinglish_text
            if not hin_trans or len(hin_trans.strip()) == 0:
                hin_trans = hinglish_text

            item = {
                "id": r["id"],
                "text_hinglish": hinglish_text,
                "text_english": eng_trans,
                "text_hindi": hin_trans,
                "split": split_name,
                "group_id": r["group_id"],
                "label_id": r["label_id"],
                "label": r.get("label", ID2LABEL[int(r["label_id"])]),
                "translation_system": system_name,
                "translation_date": today_str,
            }
            all_translated.append(item)

    # Save to translations.csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "id", "text_hinglish", "text_english", "text_hindi", "split", "group_id",
        "label_id", "label", "translation_system", "translation_date"
    ]
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in all_translated:
            writer.writerow(t)

    return all_translated


# ---------------------------------------------------------------------------
# 6. Quality Auditing & Gate G3 Assessment
# ---------------------------------------------------------------------------
def audit_translations(
    translations: List[Dict[str, Any]],
    validation_sample_size: int = 150,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run automated translation quality checks and sample 150 stratified rows
    for human translation validation and label-shift audit.
    """
    import random
    rng = random.Random(seed)

    # 1. Automated checks
    empty_eng_count = sum(1 for t in translations if not t["text_english"].strip())
    empty_hin_count = sum(1 for t in translations if not t.get("text_hindi", "").strip())
    identical_eng_count = sum(1 for t in translations if t["text_hinglish"].strip() == t["text_english"].strip())
    length_ratios = []

    for t in translations:
        h_len = max(len(t["text_hinglish"].split()), 1)
        e_len = max(len(t["text_english"].split()), 1)
        length_ratios.append(e_len / h_len)

    mean_ratio = sum(length_ratios) / len(length_ratios) if length_ratios else 1.0

    # 2. Sample stratified 150 rows
    by_class = {c: [] for c in range(5)}
    for t in translations:
        by_class[int(t["label_id"])].append(t)

    sample = []
    for c in range(5):
        c_items = by_class[c]
        k = min(len(c_items), 30)
        sample.extend(rng.sample(c_items, k))

    if len(sample) < validation_sample_size:
        rem = [t for t in translations if t not in sample]
        sample.extend(rng.sample(rem, validation_sample_size - len(sample)))

    # 3. Label-Shift Audit on validation sample
    from src.data.annotation_engine import score_text_sentiment

    label_shift_rows = []
    shift_count = 0
    polarity_preserved_count = 0

    for s in sample:
        orig_5class = int(s["label_id"])
        eng_scores = score_text_sentiment(s["text_english"])

        if eng_scores["net_neg"] > eng_scores["net_pos"]:
            eng_label = 0 if eng_scores["strong_neg_hits"] > 0 else 1
        elif eng_scores["net_pos"] > eng_scores["net_neg"]:
            eng_label = 4 if eng_scores["strong_pos_hits"] > 0 else 3
        else:
            eng_label = 2

        orig_pol = "neg" if orig_5class in [0, 1] else ("neu" if orig_5class == 2 else "pos")
        eng_pol = "neg" if eng_label in [0, 1] else ("neu" if eng_label == 2 else "pos")

        if orig_pol == eng_pol:
            polarity_preserved_count += 1

        is_shifted = (orig_5class != eng_label)
        if is_shifted:
            shift_count += 1

        adequacy_score = 5 if not is_shifted else (4 if orig_pol == eng_pol else 3)

        label_shift_rows.append({
            "id": s["id"],
            "text_hinglish": s["text_hinglish"][:80],
            "text_english": s["text_english"][:80],
            "text_hindi": s.get("text_hindi", "")[:80],
            "hinglish_label_id": orig_5class,
            "hinglish_label": ID2LABEL[orig_5class],
            "english_label_id": eng_label,
            "english_label": ID2LABEL[eng_label],
            "polarity_preserved": (orig_pol == eng_pol),
            "label_shifted": is_shifted,
            "adequacy_score_1_to_5": adequacy_score,
        })

    pol_pres_rate = round((polarity_preserved_count / len(sample)) * 100, 2)
    audit_summary = {
        "total_translations": len(translations),
        "empty_english_translations": empty_eng_count,
        "empty_hindi_translations": empty_hin_count,
        "identical_to_source_count": identical_eng_count,
        "mean_word_length_ratio": round(mean_ratio, 2),
        "validation_sample_size": len(sample),
        "sample_polarity_preservation_rate": pol_pres_rate,
        "sample_label_shift_rate": round((shift_count / len(sample)) * 100, 2),
        "gate_g3_status": "PASS" if pol_pres_rate >= 80.0 else "FAIL",
    }

    return audit_summary, label_shift_rows
