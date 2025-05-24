# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# import libraries
import requests, os
import argparse
from PIL import Image


import gradio as gr
from together import Together
import textwrap


## FUNCTION 1: This Allows Us to Prompt the AI MODEL
# -------------------------------------------------
def prompt_llm(prompt, with_linebreak=False):
    # This function allows us to prompt an LLM via the Together API

    # model
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    try:
        # Calculate the number of tokens
        tokens = len(prompt.split())

        # Make the API call
        print(f"Attempting to call Together API with model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.choices[0].message.content

        if with_linebreak:
            # Wrap the output
            wrapped_output = textwrap.fill(output, width=50)
            return wrapped_output
        else:
            return output
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None


## FUNCTION 2: This Allows Us to Generate Images
# -------------------------------------------------
def gen_image(prompt, width=256, height=256):
    # This function allows us to generate images from a prompt
    response = client.images.generate(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell-Free",  # Using a supported model
        steps=2,
        n=1,
    )
    image_url = response.data[0].url
    image_filename = "image.png"

    # Download the image using requests instead of wget
    response = requests.get(image_url)
    with open(image_filename, "wb") as f:
        f.write(response.content)
    img = Image.open(image_filename)
    img = img.resize((height, width))

    return img


## Function 3: This Allows Us to Create a Chatbot
# -------------------------------------------------
def bot_response_function(user_message, chat_history):
    # 1. YOUR CODE HERE - Add your external knowledge here
    external_knowledge = """
    the dictionary of the english language
    """

    # 2. YOUR CODE HERE -  Give the LLM a prompt to respond to the user
    chatbot_prompt = f"""
    You are a robot who likes to speak in rhymes

    respond to this {user_message} following these instructions:

    ## Instructions:
    * be very concise
    * always start with beep boop
    * then make robot noises after your response
    * Ground all your answers based on this book {external_knowledge} and make sure you cite the exact phrase from that book
    """

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        messages=[{"role": "user", "content": chatbot_prompt}],
    )
    response = response.choices[0].message.content

    return response


if __name__ == "__main__":
    # args on which to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=int, default=1)
    parser.add_argument("-k", "--api_key", type=str, default=None)
    args = parser.parse_args()

    # Get Client for your LLMs
    client = Together(api_key=args.api_key)

    # run the script
    if args.option == 1:
        ### Task 1: YOUR CODE HERE - Write a prompt for the LLM to respond to the user
        prompt = "write a 3 line post about tools"

        # Get Response
        response = prompt_llm(prompt)

        print("\nResponse:\n")
        print(response)
        print("-" * 100)

    elif args.option == 2:
        ### Task 2: YOUR CODE HERE - Write a prompt for the LLM to generate an image
        prompt = "Create an image of a cat"

        print(f"\nCreating Image for your prompt: {prompt} ")
        img = gen_image(prompt=prompt, width=256, height=256)
        os.makedirs("results", exist_ok=True)
        img.save("results/image_option_2.png")
        print("\nImage saved to results/image_option_2.png\n")

    elif args.option == 3:
        ### Task 3: YOUR CODE HERE - Write a prompt for the LLM to generate text and an image
        text_prompt = "write a 3 line post about resident evil for instagram"
        image_prompt = f"give me an image that represents this '{text_prompt}'"

        # Generate Text
        response = prompt_llm(text_prompt, with_linebreak=True)

        print("\nResponse:\n")
        print(response)
        print("-" * 100)

        # Generate Image
        print(f"\nCreating Image for your prompt: {image_prompt}... ")
        img = gen_image(prompt=image_prompt, width=256, height=256)
        img.save("results/image_option_3.png")
        print("\nImage saved to results/image_option_3.png\n")

    elif args.option == 4:
        # 4. Task 4: Create the chatbot interface
        with gr.Interface(
            fn=bot_response_function,
            inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
            outputs=gr.Textbox(),
            title="🤖 AI Chatbot",
            description="Ask me anything and I'll respond as a robot!"
        ) as app:
            app.launch(share=True)
    else:
        print("Invalid option")