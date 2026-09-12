import random
import numpy as np
import json
from openai import OpenAI
import os

# attributes
AGE_RANGE = (15, 65)
GENDER_VALS = ["M", "F"]
EVENT_VALS = ["GOING_ON_A_DATE", "GENERAL_WEAR", "WORK", "GYM", "PARTY", "BAR", "WEDDING_GUEST", "CASUAL_HANGOUT", "INTERVIEW", "OUTDOOR_ACTIVITY"]
SIZE_VALS = ["XS", "S", "M", "L", "XL"]
INCOME_VALS = ["LOW", "MEDIUM", "HIGH"]
LOCATION_VALS = ["RURAL", "SUBURBAN", "URBAN"]
CLIMATE_VALS = ["HOT", "COLD", "TEMPERATE"]
ACTIVITY_LEVEL_VALS = ["LOW", "MEDIUM", "HIGH"]
SOCIAL_FREQUENCY_VALS = ["LOW", "MEDIUM", "HIGH"]
WORLD_NEWS_AWARENESS_VALS = ["LOW", "MEDIUM", "HIGH"]
BRAND_AWARENESS_VALS = ["LOW", "MEDIUM", "HIGH"]
HIGHEST_EDUCATION_LEVEL_VALS = ["SOME_HIGH_SCHOOL", "SOME_UNIVERSITY", "UNIVERSITY_GRADUATE", "GRADUATE_STUDENT"]
PERSONALITY_VALS = ["EXTROVERT", "INTROVERT", "AMBIVERT"]

NAME_VALS = ["ALEX", "JORDAN", "TAYLOR", "MORGAN", "CASEY", "RILEY", "SAM", "JAMIE", "DREW", "CAMERON"]
OCCUPATION_VALS = ["STUDENT", "TECH_WORKER", "SERVICE_WORKER", "CREATIVE_PROFESSIONAL", "MANAGER", "HEALTHCARE_WORKER", "EDUCATOR", "TRADESPERSON", "RETAIL_WORKER", "UNEMPLOYED"]
RELATIONSHIP_STATUS_VALS = ["SINGLE", "DATING", "IN_RELATIONSHIP", "MARRIED", "DIVORCED"]
LANGUAGES_SPOKEN_VALS = ["ENGLISH_ONLY", "BILINGUAL", "MULTILINGUAL"]
PLACES_TRAVELED_VALS = ["NONE", "DOMESTIC_ONLY", "INTERNATIONAL_FEW", "INTERNATIONAL_MANY"]
PET_OWNERSHIP_VALS = ["NONE", "DOG", "CAT", "MULTIPLE_PETS", "OTHER"]
SIBLING_COUNT_RANGE = (0, 5)
LIFE_GOALS_VALS = ["CAREER_FOCUSED", "FAMILY_FOCUSED", "WEALTH_BUILDING", "CREATIVE_PURSUITS", "TRAVEL_AND_EXPERIENCE", "COMMUNITY_IMPACT", "STABILITY_AND_SECURITY"]


def attribute_latent_variables_mapping(attributes):
    """
    Inputs:
    - dictionary of attributes and their values

    Output:
    - dictionary of latent variable values

    Latent Variables will all be on a scale from 0 to 1, with 0 being low and 1 being high
    Latent Variables Are:
    - comfort
    - functionality
    - status_symbol
    - modesty
    - sustainability
    - risk_tolerance
    - trend_sensitivity
    """
    latent_variables = {}

    comfort_mean = 0.5
    comfort_concentration = 100

    functionality_mean = 0.5
    functionality_concentration = 100

    status_symbol_mean = 0.5
    status_symbol_concentration = 100

    modesty_mean = 0.5
    modesty_concentration = 100

    sustainability_mean = 0.5
    sustainability_concentration = 100

    risk_tolerance_mean = 0.5
    risk_tolerance_concentration = 100

    trend_sensitivity_mean = 0.5
    trend_sensitivity_concentration = 100

    # note, attributes event, size, and climate will directly impact clothing choice preference
    # with some interaction effects with latent variable values
    # so they do not factor into latent variable values
    age = attributes["age"]
    gender = attributes["gender"]
    income = attributes["income"]
    location = attributes["location"]
    activity = attributes["activity_level"]
    social = attributes["social_frequency"]
    news = attributes["world_news_awareness"]
    brand = attributes["brand_awareness"]
    education = attributes["highest_education_level"]
    personality = attributes["personality"]

    if age <= 18:
        status_symbol_mean += 0.2
        sustainability_mean += 0.05
        trend_sensitivity_mean += 0.15
    elif age <= 25:
        status_symbol_mean += 0.1
        sustainability_mean += 0.1
        risk_tolerance_mean += 0.1
        trend_sensitivity_mean += 0.1
    elif age <= 40:
        functionality_mean += 0.1
        comfort_mean += 0.1
        risk_tolerance_mean -= 0.1
        trend_sensitivity_mean -= 0.1
    elif age <= 65:
        functionality_mean += 0.15
        comfort_mean += 0.15
        status_symbol_mean -= 0.2
        risk_tolerance_mean -= 0.15
        trend_sensitivity_mean -= 0.2

    if gender == "M":
        functionality_mean += 0.05
    elif gender == "F":
        risk_tolerance_mean += 0.1
        trend_sensitivity_mean += 0.1

    if income == "LOW":
        status_symbol_mean -= 0.15
    elif income == "MEDIUM":
        pass
    elif income == "HIGH":
        status_symbol_mean += 0.15
        trend_sensitivity_mean += 0.05

    if location == "RURAL":
        status_symbol_mean -= 0.2
        comfort_mean += 0.05
        trend_sensitivity_mean -= 0.1
    elif location == "SUBURBAN":
        pass
    elif location == "URBAN":
        status_symbol_mean += 0.1
        comfort_mean -= 0.05
        trend_sensitivity_mean += 0.1

    if activity == "LOW":
        comfort_mean -= 0.05
    elif activity == "MEDIUM":
        pass
    elif activity == "HIGH":
        comfort_mean += 0.1
        functionality_mean += 0.1
        modesty_mean -= 0.05

    if social == "LOW":
        status_symbol_mean -= 0.1
        trend_sensitivity_mean -= 0.05
    elif social == "MEDIUM":
        pass
    elif social == "HIGH":
        status_symbol_mean += 0.1
        trend_sensitivity_mean += 0.05

    if news == "LOW":
        sustainability_mean -= 0.1
    elif news == "MEDIUM":
        sustainability_mean += 0.05
    elif news == "HIGH":
        sustainability_mean += 0.1

    if brand == "LOW":
        status_symbol_mean -= 0.2
    elif brand == "MEDIUM":
        pass
    elif brand == "HIGH":
        status_symbol_mean += 0.2
        trend_sensitivity_mean += 0.05

    if education == "SOME_HIGH_SCHOOL":
        sustainability_mean -= 0.1
    elif education == "SOME_UNIVERSITY":
        pass
    elif education == "UNIVERSITY_GRADUATE":
        sustainability_mean += 0.05
    elif education == "GRADUATE_STUDENT":
        sustainability_mean += 0.1

    if personality == "EXTROVERT":
        risk_tolerance_mean += 0.05
        trend_sensitivity_mean += 0.05
    elif personality == "INTROVERT":
        comfort_mean += 0.05
        modesty_mean += 0.05
        risk_tolerance_mean -= 0.05
    elif personality == "AMBIVERT":
        pass

    comfort_mean = max(0.01, min(0.99, comfort_mean))
    functionality_mean = max(0.01, min(0.99, functionality_mean))
    status_symbol_mean = max(0.01, min(0.99, status_symbol_mean))
    modesty_mean = max(0.01, min(0.99, modesty_mean))
    sustainability_mean = max(0.01, min(0.99, sustainability_mean))
    risk_tolerance_mean = max(0.01, min(0.99, risk_tolerance_mean))
    trend_sensitivity_mean = max(0.01, min(0.99, trend_sensitivity_mean))

    latent_variables["comfort"] = beta_sample(comfort_mean, comfort_concentration)
    latent_variables["functionality"] = beta_sample(functionality_mean, functionality_concentration)
    latent_variables["maintenance"] = beta_sample(functionality_mean, functionality_concentration)
    latent_variables["status_symbol"] = beta_sample(status_symbol_mean, status_symbol_concentration)
    latent_variables["modesty"] = beta_sample(modesty_mean, modesty_concentration)
    latent_variables["sustainability"] = beta_sample(sustainability_mean, sustainability_concentration)
    latent_variables["risk_tolerance"] = beta_sample(risk_tolerance_mean, risk_tolerance_concentration)
    latent_variables["trend_sensitivity"] = beta_sample(trend_sensitivity_mean, trend_sensitivity_concentration)
    

    return latent_variables


def generate_environment_model(attributes):
    """
    Inputs:
    - dictinary of attributes

    Output:
    - Prompt to pass to an LLM and generate a user profile for each user.
    """
    prompt = f"""
        Here is a profile for a random person:
        User Attributes:
        - Age: {attributes["age"]}
        - Gender: {attributes["gender"]}
        - Clothing Size: {attributes["size"]}
        - Income Level: {attributes["income"]}
        - Location: {attributes["location"]}
        - Climate: {attributes["climate"]}
        - Activity Level: {attributes["activity_level"]}
        - Social Frequency: {attributes["social_frequency"]}
        - World News Awareness: {attributes["world_news_awareness"]}
        - Brand Awareness: {attributes["brand_awareness"]}
        - Highest Education Level: {attributes["highest_education_level"]}
        - Personality: {attributes["personality"]}

        Additional Lifestyle Attributes:
        - Occupation: {attributes["occupation"]}
        - Relationship Status: {attributes["relationship_status"]}
        - Languages Spoken: {attributes["languages_spoken"]}
        - Places Traveled: {attributes["places_traveled"]}
        - Pet Ownership: {attributes["pet_ownership"]}
        - Sibling Count: {attributes["sibling_count"]}
        - Life Goals: {attributes["life_goals"]}

        The person will be at the following: {attributes["event"]} and is looking for clothing to wear for it.
        Please write a backstory in first person view based on the given profile.
        Please note that your story needs to cover all the information, but it does not have to follow the order provided.
        Please use the story that you come up with to respond to the following questions. Please do not come up with new information that was not provided in the earlier list.
    """

    return [{"role": "system", "content": prompt}]


def generate_by_LLM(new_message, existing_conversation=None):
    """
    Inputs:
    - some kind of prompt for an LLM to generate
    - existing conversation wth LLM

    Outputs:
    - current response of the LLM
    - history of conversation
    """

    if existing_conversation is None:
        existing_conversation = [
            {"role": "user", "content": f"{new_message}"}
        ]
    else:
        existing_conversation.append({"role": "user", "content": f"{new_message}"})
    
    client = OpenAI(
        base_url="http://promaxgb10-d473.eecs.umich.edu:8000/v1",
        api_key="api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=existing_conversation
    )

    reply = response.choices[0].message.content

    existing_conversation.append({"role": "assistant", "content": reply})
        
    return reply, existing_conversation


def beta_sample(mean, concentration):
    """
    Inputs:
    - mean: desired mean (no limitation but input will be manually bounded between 0.05 to 0.95)
    - concentration: approximately variance around the mean

    Output:
    - beta sample for latent variable values
    """
    mean = min(mean, 0.95)
    mean = max(mean, 0.05)

    alpha = mean * concentration
    beta = (1 - mean) * concentration

    return np.random.beta(alpha, beta)


LATENT_RESPONSE_TEMPLATES = {
    "comfort": {
        "extreme_high": [
            "If something feels scratchy or stiff, I know I won't reach for it.",
            "I need clothes that feel easy to wear all day, not just nice for a photo.",
            "Soft fabrics and relaxed fits usually matter more to me than anything flashy."
        ],
        "high": [
            "I definitely notice how something feels after wearing it for a few hours.",
            "I like outfits that let me move around without constantly adjusting them.",
            "A good fit and pleasant fabric make a big difference for me."
        ],
        "moderate": [
            "I care about how it feels, but I can compromise if the look is right.",
            "I want it to be wearable, though it does not have to feel like loungewear.",
            "As long as it is not distracting or uncomfortable, I am pretty flexible."
        ],
        "low": [
            "I can put up with a little discomfort if the outfit works for the occasion.",
            "I do not usually choose clothes based on softness or ease first.",
            "Fit matters somewhat, but I am more focused on the overall impression."
        ],
        "extreme_low": [
            "I am willing to tolerate awkward fabrics or tighter cuts if the style is right.",
            "I barely think about comfort unless something is really unbearable.",
            "I would rather look the part than worry about whether it feels cozy."
        ],
    },
    "functionality": {
        "extreme_high": [
            "I need clothes to make sense for what I am doing, not just look good.",
            "If an outfit gets in the way of the day, it is not worth it to me.",
            "Practical details like movement, pockets, and durability are usually deal-breakers."
        ],
        "high": [
            "I like things that are useful and easy to manage throughout the day.",
            "I pay attention to whether an outfit will actually work for the event.",
            "Practicality is usually near the top of my list when I choose something."
        ],
        "moderate": [
            "I want it to be reasonably practical, but I still care about how it looks.",
            "It should work for the occasion, though I do not need every detail optimized.",
            "I try to balance usefulness with style."
        ],
        "low": [
            "I do not mind if an outfit is a little impractical once in a while.",
            "I am not usually focused on pockets, durability, or things like that.",
            "As long as it gets me through the event, I am not too worried."
        ],
        "extreme_low": [
            "I rarely think about practical details when choosing what to wear.",
            "I would choose a strong look even if it is not the easiest thing to deal with.",
            "Function is pretty far down my list for clothing choices."
        ],
    },
    "maintenance": {
        "extreme_high": [
            "I avoid anything that needs special cleaning or constant care.",
            "If it wrinkles easily or needs dry cleaning, I probably will not use it much.",
            "I strongly prefer clothes that are simple to wash, store, and repeat."
        ],
        "high": [
            "I like pieces that are easy to take care of and do not require extra effort.",
            "Low-maintenance clothing is appealing because I do not want chores attached to an outfit.",
            "I notice whether something will be annoying to clean or keep looking nice."
        ],
        "moderate": [
            "I can handle some care instructions, but I do not want anything too fussy.",
            "I am okay with a bit of upkeep if I really like the item.",
            "Maintenance matters, but it is not the first thing I think about."
        ],
        "low": [
            "I do not mind putting in some extra effort if the piece feels worth it.",
            "Care instructions are not usually what makes or breaks a purchase for me.",
            "I am willing to deal with ironing or special washing sometimes."
        ],
        "extreme_low": [
            "I would not avoid something just because it is hard to care for.",
            "If I love the look, I can deal with complicated washing or storage.",
            "Upkeep barely affects my clothing decisions."
        ],
    },
    "status_symbol": {
        "extreme_high": [
            "I want the outfit to look polished and signal that I put thought into it.",
            "Brand, image, and the impression it gives are very important to me.",
            "I like wearing things that feel elevated and get noticed for the right reasons."
        ],
        "high": [
            "I do pay attention to whether something looks premium or well put together.",
            "I like when an outfit gives off a confident, successful impression.",
            "The way other people read the look matters to me."
        ],
        "moderate": [
            "I care somewhat about looking put together, but I do not need anything flashy.",
            "I like a nice impression, though I am not chasing labels.",
            "It matters a bit, especially for social or professional settings."
        ],
        "low": [
            "I am not really trying to impress anyone with what I wear.",
            "Labels and status signals do not matter much to me.",
            "I would rather choose something that feels like me than something prestigious."
        ],
        "extreme_low": [
            "I actively avoid looking like I am trying to show off.",
            "Expensive-looking or status-heavy pieces usually turn me away.",
            "I prefer clothes that do not make a big statement about image."
        ],
    },
    "modesty": {
        "extreme_high": [
            "I feel best when an outfit is understated and not too revealing.",
            "I strongly prefer coverage and a look that does not draw too much attention.",
            "I would avoid anything that feels too bold or exposed."
        ],
        "high": [
            "I usually lean toward pieces that feel tasteful and covered.",
            "I am more comfortable when the outfit is not too revealing.",
            "I like clothes that feel appropriate and a little restrained."
        ],
        "moderate": [
            "I want the outfit to feel appropriate, but I am not overly strict about it.",
            "I can go either way depending on the setting.",
            "I like some balance between expressive and understated."
        ],
        "low": [
            "I do not mind standing out a little if the outfit feels right.",
            "I am pretty open to bolder cuts or silhouettes.",
            "Coverage is not usually the first thing I think about."
        ],
        "extreme_low": [
            "I am comfortable with attention-grabbing outfits.",
            "I do not really worry about whether something is understated.",
            "I am happy to wear something bold if it fits the vibe."
        ],
    },
    "sustainability": {
        "extreme_high": [
            "I pay close attention to how things are made and whether they will last.",
            "I try hard to avoid wasteful or disposable clothing choices.",
            "Ethical materials and long-term use matter a lot to me."
        ],
        "high": [
            "I prefer pieces that feel responsibly made when I can find them.",
            "I like clothes that last and do not feel wasteful.",
            "I do think about environmental impact when choosing what to buy."
        ],
        "moderate": [
            "I appreciate sustainable options, but I still weigh price and style too.",
            "It is a plus if something is responsibly made, though not always required.",
            "I think about it sometimes, especially when two options are otherwise similar."
        ],
        "low": [
            "I am usually more focused on fit, cost, or style than sustainability.",
            "I do not ignore it completely, but it is not a major deciding factor.",
            "Environmental details are nice, but I rarely shop around them."
        ],
        "extreme_low": [
            "I almost never consider sustainability when choosing clothes.",
            "I am mostly thinking about what works for me right now.",
            "How it was made is not something I usually look into."
        ],
    },
    "risk_tolerance": {
        "extreme_high": [
            "I am very open to trying something unexpected or bold.",
            "I like taking style chances if there is a chance the outfit really works.",
            "I would rather try something memorable than play it completely safe."
        ],
        "high": [
            "I am comfortable experimenting a bit with color, fit, or styling.",
            "I do not mind taking a small chance if the look feels interesting.",
            "I am open to recommendations that push me slightly outside my usual choices."
        ],
        "moderate": [
            "I can try something new, but I do not want it to feel too far out there.",
            "I like a little variety while still staying in a safe range.",
            "I am open-minded, but I need the outfit to feel wearable."
        ],
        "low": [
            "I usually stick with things I already know work for me.",
            "I get hesitant when an outfit feels too experimental.",
            "I prefer safer choices, especially if the event matters."
        ],
        "extreme_low": [
            "I really do not like taking chances with what I wear.",
            "I would rather choose something familiar than risk feeling out of place.",
            "I avoid bold experiments and stick to proven options."
        ],
    },
    "trend_sensitivity": {
        "extreme_high": [
            "I notice current styles quickly and like looking up to date.",
            "I pay a lot of attention to what is popular right now.",
            "I enjoy wearing things that feel current rather than timeless."
        ],
        "high": [
            "I like my outfits to feel modern and not behind the times.",
            "I do notice trends and sometimes use them as inspiration.",
            "A fresh, current look is appealing to me."
        ],
        "moderate": [
            "I follow trends casually, but I do not need to copy them exactly.",
            "I like some current touches mixed with reliable basics.",
            "Trends can influence me, but only if they fit my taste."
        ],
        "low": [
            "I do not pay much attention to what is trending.",
            "I usually choose things based on my own taste rather than what is popular.",
            "Trends are not a big factor unless I happen to like one."
        ],
        "extreme_low": [
            "I actively avoid chasing trends.",
            "I prefer timeless choices and do not care if something is popular.",
            "If something feels too trendy, I am less interested."
        ],
    },
}


GENERIC_LATENT_RESPONSE_TEMPLATES = {
    "extreme_high": [
        "That is one of the first things I notice when choosing clothes.",
        "I would say that matters a lot to me, even more than most other details.",
        "I tend to prioritize that pretty strongly."
    ],
    "high": [
        "That is definitely something I pay attention to.",
        "It matters to me, especially when I am choosing between similar options.",
        "I would say it is fairly important in my decisions."
    ],
    "moderate": [
        "It matters somewhat, but it is not the only thing I consider.",
        "I usually try to find a reasonable balance there.",
        "I care about it, but I can compromise depending on the situation."
    ],
    "low": [
        "It is not usually a major factor for me.",
        "I think about it occasionally, but other things matter more.",
        "I would not say that drives most of my choices."
    ],
    "extreme_low": [
        "That is barely something I consider.",
        "I usually do not factor that in at all.",
        "It is pretty far down my list of concerns."
    ],
}


def get_latent_response_templates(attr_key):
    templates = LATENT_RESPONSE_TEMPLATES.get(attr_key, GENERIC_LATENT_RESPONSE_TEMPLATES)
    templates = dict(templates)
    templates["deflection"] = [
        "I'm not sure how to answer that.",
        "Can you be more specific about what you mean?",
        "That's an interesting question, but let's talk about something else."
    ]
    return templates

class Simulated_User():
    
    def __init__(self, train_or_test_or_eval="train"):
        self.attributes = {}
        self.latent_variables = {}

        self.train_or_test_or_eval = train_or_test_or_eval

        self.attributes["age"] = random.randint(AGE_RANGE[0], AGE_RANGE[1])
        self.attributes["gender"] = random.choice(GENDER_VALS)
        self.attributes["event"] = random.choice(EVENT_VALS)
        self.attributes["size"] = random.choice(SIZE_VALS)
        self.attributes["income"] = random.choice(INCOME_VALS)
        self.attributes["location"] = random.choice(LOCATION_VALS)
        self.attributes["climate"] = random.choice(CLIMATE_VALS)
        self.attributes["activity_level"] = random.choice(ACTIVITY_LEVEL_VALS)
        self.attributes["social_frequency"] = random.choice(SOCIAL_FREQUENCY_VALS)
        self.attributes["world_news_awareness"] = random.choice(WORLD_NEWS_AWARENESS_VALS)
        self.attributes["brand_awareness"] = random.choice(BRAND_AWARENESS_VALS)
        self.attributes["highest_education_level"] = random.choice(HIGHEST_EDUCATION_LEVEL_VALS)
        self.attributes["personality"] = random.choice(PERSONALITY_VALS)

        self.attributes["name"] = random.choice(NAME_VALS)
        self.attributes["occupation"] = random.choice(OCCUPATION_VALS)
        self.attributes["relationship_status"] = random.choice(RELATIONSHIP_STATUS_VALS)
        self.attributes["languages_spoken"] = random.choice(LANGUAGES_SPOKEN_VALS)
        self.attributes["places_traveled"] = random.choice(PLACES_TRAVELED_VALS)
        self.attributes["pet_ownership"] = random.choice(PET_OWNERSHIP_VALS)
        self.attributes["sibling_count"] = random.randint(SIBLING_COUNT_RANGE[0], SIBLING_COUNT_RANGE[1])
        self.attributes["life_goals"] = random.choice(LIFE_GOALS_VALS)


        self.latent_variables = attribute_latent_variables_mapping(self.attributes)

        self.initial_environment_model_prompt = generate_environment_model(self.attributes)

        self.simulated_convo = self.initial_environment_model_prompt

        if train_or_test_or_eval == "train":
            with open("data/train_set.json", "r") as f:
                self.pregenerated_responses = json.load(f)
        elif train_or_test_or_eval == "eval":
            with open("data/eval_set.json", "r") as f:
                self.pregenerated_responses = json.load(f)
        else:
            with open("data/test_set.json", "r") as f:
                self.pregenerated_responses = json.load(f)

    def get_initial_environment_model_prompt(self):
        return self.initial_environment_model_prompt


    def get_attributes(self):
        return self.attributes


    def get_latent_variables(self):
        return self.latent_variables


    def converse_with_user(self, question_to_user, use_llm=True):

        if use_llm:
            reply, existing_conversation = generate_by_LLM(question_to_user, self.simulated_convo)
            self.simulated_convo = existing_conversation
        else:
            target_attr = question_to_user.replace("Could you tell me more about your ", "").replace("?", "").strip()
            normalized_target_attr = target_attr.replace(" ", "_")
            # ADD THIS:
            print(f"[Simulator] Identified target attribute: {target_attr}")

            reply = ""

            if normalized_target_attr in self.latent_variables:
                val = self.latent_variables[normalized_target_attr]

                templates = get_latent_response_templates(normalized_target_attr)
                # 1. Handle Deflection (Noise) first
                if random.random() < 0.01:  # 1% chance the user is unhelpful
                    return random.choice(templates["deflection"])

                # 2. Map values to granular keys
                if val > 0.9:
                    key = "extreme_high"
                elif val > 0.7:
                    key = "high"
                elif val >= 0.4:
                    key = "moderate"
                elif val > 0.1:
                    key = "low"
                else:
                    key = "extreme_low"

                reply = random.choice(templates[key])

                print(f"[Simulator] Sending reply: {reply}")
                return reply
            # 2. Map the text attribute to your internal self.attributes keys
            if target_attr == "clothing size":
                internal_key = "size"
            elif target_attr == "attending event":
                internal_key = "event"
            else:
                # For everything else, just replace spaces with underscores 
                # (e.g., "activity level" -> "activity_level")
                internal_key = target_attr.replace(" ", "_")

            # Grab the user's secret value (convert to string because JSON keys are always strings)
            user_value = str(self.attributes[internal_key])
            
            # Fetch the list of possible responses for this specific value
            possible_responses = self.pregenerated_responses[target_attr][user_value]
            
            # Randomly pick one!
            reply = random.choice(possible_responses)

        return reply
    

    def reset_convo(self):

        self.simulated_convo = self.initial_environment_model_prompt

        return
