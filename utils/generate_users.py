import random
import numpy as np
import json
from openai import OpenAI

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
