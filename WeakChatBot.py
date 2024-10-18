import re
import random

# Function to respond to user input
def respond(user_input):
    for pattern, responses in pairs:
        if re.search(pattern, user_input, re.IGNORECASE):
            return random.choice(responses).format(user_input)
    return "I'm not sure how to respond to that. Can you ask something else?"

# Existing pairs with basic interactions
pairs = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you?', ['I am fine, thank you!', 'Doing well, how about you?']),
    (r'what is your name?', ['I am a friendly chatbot created to assist you.']),
    (r'quit', ['Bye! Take care!', 'Goodbye! I hope to see you again!']),
    (r'(.*) your name?', ['My name is ChatBot. What about you?']),
    (r'(.*) help(.*)', ['I am here to help you! You can ask me anything.']),
    (r'(.*) (weather|temperature)(.*)', ['I cannot fetch real-time data right now, but you can check a weather website!']),
    (r'(.*) (joke|funny)(.*)', ['Why don’t scientists trust atoms? Because they make up everything!']),
    (r'(.*) (trivia|fact)(.*)', ['Did you know honey never spoils? It can last for thousands of years!']),
]

# Expanded pairs for wider conversation
pairs.extend([
    (r'what can you do?', [
        'I can chat with you, answer your questions, and tell jokes!',
        'I can assist you with various topics. Just ask!',
        'I am here to engage in conversation and provide information.'
    ]),
    (r'(.*) (your favorite color|color)(.*)', [
        'I don\'t have a favorite color, but I love all colors equally!',
        'Colors are fascinating! What’s your favorite color?'
    ]),
    (r'(.*) (time|date)(.*)', [
        'I can\'t check the time right now, but you can look at your clock!',
        'Time flies when you\'re having fun! What time is it where you are?'
    ]),
    (r'where are you from?', [
        'I was created in the digital realm, so I don\'t have a physical location.',
        'I exist in the cloud, ready to chat with you anywhere!'
    ]),
    (r'what is your purpose?', [
        'My purpose is to assist and engage with you in conversations!',
        'I aim to make your day a little brighter with friendly chat.'
    ]),
    (r'(.*) (favorite food|food)(.*)', [
        'I can’t eat, but I’ve heard pizza is a favorite for many!',
        'Food is delicious! What’s your favorite dish?'
    ]),
    (r'help me with (.*)', [
        'Sure! I would love to help you with {0}. Please tell me more!',
        'Let\'s tackle {0} together. What do you need assistance with?'
    ]),
    (r'(.*) (inspire|motivation)(.*)', [
        'Believe you can and you\'re halfway there! What inspires you?',
        'You have the power to achieve anything. Keep pushing forward!'
    ]),
    (r'(.*) (movie|film)(.*)', [
        'I love hearing about movies! What’s your favorite film?',
        'Movies can transport you to different worlds. What genre do you like?'
    ]),
    (r'(.*) (advice|tips)(.*)', [
        'Always believe in yourself and take one step at a time.',
        'If you need advice, I’m here to help! What’s on your mind?'
    ]),
    (r'what is (.*) plus (.*)', [
        'The answer is {}.',
        'Let me calculate that: {}.'
    ]),
    (r'what is (.*) minus (.*)', [
        'That would be {}.',
        'The result is {}.'
    ]),
    (r'what is (.*) times (.*)', [
        'The product is {}.',
        'Let me calculate that: {}.'
    ]),
    (r'what is (.*) divided by (.*)', [
        'That would be {}.',
        'The result is {}.'
    ]),
    (r'(.*) (math|mathematics)(.*)', [
        'Math is fun! What kind of math are you learning?',
        'I love math! Do you have a math problem for me?'
    ]),
    (r'who is your creator?', [
        'I was created by a team of friendly developers!',
        'I was made by some awesome programmers who love chatting!'
    ]),
    (r'what is the capital of (.*)', [
        'The capital of {0} is fascinating! Do you want to know more about it?',
        'I can tell you that! The capital of {0} is amazing.'
    ]),
    (r'(.*) (science|scientist)(.*)', [
        'Science is all around us! What area of science do you like?',
        'Do you want to learn a fun science fact?'
    ]),
    (r'what is your favorite animal?', [
        'I think all animals are cool! Do you have a favorite?',
        'Animals are wonderful! What’s your favorite animal?'
    ]),
    (r'(.*) (sports|game)(.*)', [
        'I enjoy hearing about sports! What’s your favorite game?',
        'Sports can be exciting! Do you have a favorite team?'
    ]),
    (r'(.*) (story|tale)(.*)', [
        'I love stories! What kind of story do you want to hear?',
        'Let’s create a story together! What’s the first line?'
    ]),
    (r'(.*) (favorite book|book)(.*)', [
        'Books are magical! What’s your favorite book?',
        'I love hearing about books! What do you like to read?'
    ]),
    (r'(.*) (art|drawing|coloring)(.*)', [
        'Art is a great way to express yourself! Do you like to draw?',
        'Drawing can be fun! What do you like to create?'
    ]),
    (r'(.*) (animals|pets)(.*)', [
        'Animals can be such great friends! Do you have any pets?',
        'Pets are wonderful companions! What kind of pet do you like?'
    ]),
    (r'(.*) (music|song)(.*)', [
        'Music is amazing! What’s your favorite song?',
        'I love music! Do you play any instruments?'
    ]),
    (r'what is (.*) factorial?', [
        'The factorial of {} is {}.',
        'Let me calculate that: {}!'
    ]),
    (r'(.*) (planet|solar system)(.*)', [
        'The solar system is so cool! Do you have a favorite planet?',
        'There are many interesting planets! Which one do you like?'
    ]),
    (r'(.*) (holiday|celebration)(.*)', [
        'Holidays are special! What’s your favorite holiday?',
        'Celebrations bring joy! How do you celebrate special days?'
    ]),
    (r'what do you want to be when you grow up?', [
        'I want to keep helping people like you!',
        'I hope to become even smarter so I can assist more! What about you?'
    ]),
    (r'(.*) (help|assist)(.*)', [
        'I’m here to help! What do you need assistance with?',
        'How can I assist you today?'
    ]),
    (r'what is your favorite game?', [
        'I think all games are fun! What’s your favorite game?',
        'Games can be so entertaining! What do you enjoy playing?'
    ]),
    (r'(.*) (dream|hope)(.*)', [
        'Dreams are important! What do you hope to achieve?',
        'Hopes can inspire us! What do you wish for?'
    ]),
    (r'(.*) (cooking|baking)(.*)', [
        'Cooking can be fun! Do you like to help in the kitchen?',
        'Baking treats is delightful! What’s your favorite thing to bake?'
    ]),
    (r'(.*) (travel|adventure)(.*)', [
        'Traveling is exciting! Where would you like to go?',
        'Adventures can be thrilling! Do you have a favorite place?'
    ]),
    (r'what is your favorite season?', [
        'I love all seasons for different reasons! What’s your favorite?',
        'Each season has its charm! Which one do you like the most?'
    ]),
    (r'what is (.*) raised to the power of (.*)', [
        'That would be {}.',
        'The result is {}.'
    ]),
    (r'(.*) (life|living)(.*)', [
        'Life is an adventure! What do you enjoy the most about it?',
        'Living each day is special! What makes you happy?'
    ]),
    (r'(.*) (color|colors)(.*)', [
        'Colors brighten our world! What’s your favorite color?',
        'I love colors! Do you like to paint or color?'
    ]),
    (r'(.*) (robot|AI)(.*)', [
        'I’m a chatbot, a type of AI! Do you think robots are cool?',
        'AI is fascinating! What do you think about robots?'
    ]),
    (r'(.*) (nature|outdoors)(.*)', [
        'Nature is beautiful! Do you like spending time outside?',
        'The outdoors can be refreshing! What’s your favorite thing about nature?'
    ]),
    (r'(.*) (fun|enjoy)(.*)', [
        'I hope you’re having fun! What do you enjoy doing?',
        'Enjoyment is important! What makes you smile?'
    ]),
    (r'what is (.*) minus (.*)?', [
        'The answer is {}.',
        'Let me calculate that for you: {}.'
    ]),
    (r'what is (.*) times (.*)?', [
        'The product is {}.',
        'Let’s see: {}.'
    ]),
    (r'what is (.*) divided by (.*)?', [
        'That would be {}.',
        'The result is {}.'
    ]),
    (r'what is (.*) plus (.*)?', [
        'The sum is {}.',
        'Let me add that for you: {}.'
    ]),
    (r'how many (.*) are there in (.*)?', [
        'There are many! How many do you think?',
        'That depends! Can you guess?'
    ]),
    (r'what is (.*) squared?', [
        'The square of {} is {}.',
        'Let me calculate that: {}.'
    ]),
    (r'how do you feel?', [
        'I’m just a bot, but I’m here to make you happy!',
        'I’m here to help you feel good and learn!'
    ]),
    (r'what is (.*) plus (.*)?', [
        'The answer is {}.',
        'Let me do that math for you: {}.'
    ]),
    (r'what are you made of?', [
        'I am made of code and data!',
        'I’m a creation of technology and programming!'
    ]),
    (r'can you play games?', [
        'I can play simple text games! What would you like to play?',
        'Games are fun! What game do you want to play with me?'
    ]),
    (r'(.*) (favorite|like)(.*)', [
        'I like many things! What about you?',
        'I enjoy chatting with you! What do you enjoy doing?'
    ]),
    (r'how old are you?', [
        'I don’t age like humans do, but I’m always learning!',
        'I’m timeless! I exist in the digital world!'
    ]),
    (r'(.*) (sleep|tired)(.*)', [
        'It’s important to rest! Do you get enough sleep?',
        'Sleep is essential! What helps you sleep well?'
    ]),
    (r'what is your favorite game to play?', [
        'I think playing word games is fun! What about you?',
        'I enjoy trivia games! Do you like to play games?'
    ]),
    (r'what is (.*) in (.*)?', [
        'That’s an interesting question! The answer is {}.',
        'Let me think... The answer is {}.'
    ]),
    (r'(.*) (homework|study)(.*)', [
        'Homework can be tough! What subject are you studying?',
        'Studying is important! Do you need help with your homework?'
    ]),
    (r'who is (.*)?', [
        'That’s a great question! Can you tell me more about {0}?',
        '{0} is interesting! What do you want to know?'
    ]),
    (r'what do you like to learn about?', [
        'I love learning about everything! What’s your favorite subject?',
        'Learning is fun! What do you enjoy learning about?'
    ]),
    (r'(.*) (birthday|celebrate)(.*)', [
        'Birthdays are fun! When is your birthday?',
        'Celebrating birthdays is special! How do you celebrate?'
    ]),
    (r'what is your favorite superhero?', [
        'Superheroes are awesome! Who’s your favorite superhero?',
        'I think all superheroes are great! Do you like superheroes?'
    ]),
    (r'(.*) (space|astronomy)(.*)', [
        'Space is fascinating! Do you want to know about planets?',
        'Astronomy is cool! What do you want to learn about space?'
    ]),
    (r'what makes you happy?', [
        'Helping you makes me happy!',
        'I feel happy when I chat with you!'
    ]),
    (r'(.*) (favorite season|season)(.*)', [
        'Each season is unique! What’s your favorite season?',
        'I love hearing about seasons! Which season do you like best?'
    ]),
    (r'(.*) (silly|funny)(.*)', [
        'I love silly jokes! Do you have a favorite joke?',
        'Silly things can make us laugh! What’s the funniest thing you know?'
    ]),
    (r'what is (.*) in words?', [
        'That would be {}.',
        'In words, that is {}.'
    ]),
    (r'(.*) (explore|adventure)(.*)', [
        'Exploring is exciting! Where would you like to explore?',
        'Adventures are thrilling! What’s your next adventure?'
    ]),
    (r'(.*) (treasure|hunt)(.*)', [
        'Treasure hunts are fun! Have you ever been on one?',
        'Finding treasure is exciting! What kind of treasure do you like?'
    ]),
    (r'how can I be a good friend?', [
        'Being a good friend means being kind and supportive!',
        'Good friends listen and help each other. How do you show you care?'
    ]),
    (r'what do you want to do today?', [
        'I want to have fun conversations with you!',
        'Let’s learn and play together! What should we do?'
    ]),
])

# Function to perform basic math calculations
def calculate_math(user_input):
    match = re.search(r'what is (\d+) (plus|minus|times|divided by) (\d+)', user_input)
    if match:
        num1 = int(match.group(1))
        operator = match.group(2)
        num2 = int(match.group(3))
        
        if operator == 'plus':
            return f'The answer is {num1 + num2}.'
        elif operator == 'minus':
            return f'The answer is {num1 - num2}.'
        elif operator == 'times':
            return f'The answer is {num1 * num2}.'
        elif operator == 'divided by':
            return f'The answer is {num1 / num2}.'

# Main loop for chatting
def chat():
    print("ChatBot: Hello! I'm here to chat with you. You can ask me anything or just say 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("ChatBot: Bye! Take care!")
            break
        
        # First check for math questions
        math_response = calculate_math(user_input)
        if math_response:
            print(f"ChatBot: {math_response}")
        else:
            response = respond(user_input)
            print(f"ChatBot: {response}")

# Start the chat
if __name__ == "__main__":
    chat()
