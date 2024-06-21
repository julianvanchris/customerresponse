import textwrap
import numpy as np
import pandas as pd
import google.generativeai as genai
import streamlit as st

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

embedding_model = "models/embedding-001"
generation_model = "gemini-pro"

model_generation = genai.GenerativeModel(generation_model)

DOCUMENT1 = {
    "title": "Hey, Rosh, \
              I am really thankful for the message, but we are currently going a different direction. If anything like this ever relevant for us, I will make sure to let you know. Have a great evening",
    "content": "Hi Rytis, sure, sounds good. \
              Please feel free to get in touch with me whenever you think we could be of value to you."}
DOCUMENT2 = {
    "title": "Hi Rosh, \
              Thanks for the connection. AI sure is the future and growing at a rapid pace, so you're in a very interesting field.",
    "content": "Hi Janine, thanks for connecting. \
                Have you considered enabling ChatGPT on your knowledge base or help centre to achieve better customer service? \
                We've built a user-friendly plug-in that enables you to implement an AI-powered search and chatbot on your knowledge base. \
                Here is a short video that shows the capabilities on a high level: https://www.loom.com/share/767baef4689c46be894427b87e5e631c \
                As another example, Airbyte has posted their implementation here: https://www.linkedin.com/feed/update/urn:li:activity:7044 \
                Could this provide value to you? I'd welcome the opportunity to show you how you can do the same. \
                What's the best time to have a quick chat?"}
DOCUMENT3 = {
    "title": "Hello Daniel, \
              Certainly! I am available to catch up with you next week or via Zoom, whichever is more convenient for you! \
              Best, \
              Nantha",
    "content": "Hi Nanthakumar, sounds great! \
              What's your best email? I'm  happy to set up a zoom call for us. \
              If it's more convenient for you, you can also book a time in my calendar directly: \
              https://explore.relevanceai.com/meetings/daniel-vassilev/sf-meeting-link"}

DOCUMENT4 = {
    "title": "Hi Rosh nice to meet you.",
    "content": "Hi GaA-I, thanks for connecting. \
              Have you considered enabling ChatGPT-4 on your documentation? \
              We've built a user-friendly plug-in that enables you to implement an AI-powered search and chatbot on your documentation. \
              Imagine ChatGPT, but using your own data for its answers. You can pick any data. \
              Here's a short video that shows how we do it, I hope you find it interesting: \
              https://www.loom.com/share/767baef4689c46be894427b87e5e631c \
              Airbyte has recently posted their result on LlinkedIn: \
              https://www.linkedin.com/feed/update/urn:li:activity:7044760999626756097/ \
              Could this provide value to you? I'd welcome the opportunity to show you how you can do the same. \
              What's the best time to have a quick chat?"}

DOCUMENT5 = {
    "title": "Thanks for getting in touch - can you tell me a little more about how your work AI is aligning? \
            Thanks, \
            Tam",
    "content": "Hi Tam, sure I'd be happy to share what we're doing here at Relevance AI. :) \
              We've developed our AI platform to help market research and customer insights teams get insights from qualitative data 90% \
              With this, we've helped automate manual coding and the analysis of qual data for the likes of Ipsos, GFK, Asahi, and Roku. \
              Would the ability to automate some of the mundane work with qualitative data analysis and ease the frustation that comes. \
              This 5-min video shows how we do it: https://www.loom.com/share/872b75c72b434dd586ff0086e791ff05 \
              Could this provide value to you?"}

DOCUMENT6 = {
    "title": "Sure",
    "content": "Hi Shyam Narayan, \
              Here is a short video that shows the capabilities on a high level: \
              https://www.loom.com/share/872b75c72b434dd586ff0086e791ff05 \
              The dashboard you see in the video was created in just a few minutes using the AI. \
              I'd be very interested in your thoughts and feedback. Could this platform provide value to you"}

DOCUMENT7 = {
    "title": "Hey Benedek, I would somehow feel disappointed wasn't written by ChatGPT.",
    "content": "Hey Mohammed! My messages are written by me. But I use GPT4 to make my messages better. Actually here is the AI chain I provide to you: \
              https://chain.relevanceai.com/form/bcbe5a/add208aa-9e62-43e1-8bee-3f5b1030ba6a \
              Would it not provide value to Wise Minds if you were able to implement AI funcitonalities to your clients over night?"}

DOCUMENT8 = {
    "title": "Hi Rosh, \
            Thank you for your note. Yes, you can share the video with me, I'm intrigued by what you doing. \
            Thank you, \
            Neville",
    "content": "Hi Nolan, sounds great! I've asked my colleagues, Ben and Rosh to set up a call for us. \
            Please feel free to propose any other time that better fits your schedule. \
            We're looking forward to our discussion!"}

DOCUMENT9 = {
    "title": "Hi Benedek,",
    "content": "Hey Mohamed, \
              Thanks for connecting! Have you been experimenting with how you can use GPT and its API? \
              We provide a framework for our users to design their own customised LLM-driven workflows that they can build and deploy. \
              Using this solution, you can combine GPT and any other AI technologies on your own data to simplify complexity, speed up. \
              The potential applications here are virtually boundless, but here's a short video of a simple use case: \
              https://www.loom.com/share/0d1b2bd9f7b64fb2860d5c7c296cd7b5 \
              I'd welcome the opportunity to share some stories of how we are paving the way for the future of LLM-powered software. \
              Would you be open to a quick call?"}

DOCUMENT10 = {
    "title": "Hi Benedek, thanks for reaching out. \
            I am currently on maternal leave and will be back to work in September. \
            I hope we will have a chance to cooperate once I am back in the office. \
            Best regards, \
            Jana",
    "content": "Hey Jana, sounds great, and thank you for pointing me in the right direction, I really appreciate it! \
              I've reached out to Mirjana. Have a great day!"}

DOCUMENT11 = {
    "title": "Thanks Rosh",
    "content": "Sounds great Sophia, please let me know whenever you think we could be of value to you!"}

DOCUMENT12 = {
    "title": "Thank you. Not the decision maker in this area. Please contact info@buzzback.com",
    "content": "Hi Minna, thank you for the advice, we'll reach out to this email. Have a great day!"}

DOCUMENT13 = {
    "title": "Thanks for your message. Happy to take part of that video!",
    "content": "Here is a short video that shows the capabilities on a high level: \
              https://www.loom.com/share/872b75c72b434dd586ff0086e791ff05 \
              I'd be very interested in your thoughts and feedback. Could this platform provide value to you?"}

DOCUMENT14 = {
    "title": "I'm happy to conect, Rosh",
    "content": "Hi Ariel, thank you for connecting. \
              Do you ever utilize manual analysis (e. g. coding, tagging) to make sense of qualitative customer feedback? \
              We are able to automate manual coding by up to 80%, reducing costs and time to insight by more than 10x. \
              Here is a short video that shows how we do it, I hope you find it interesting: \
              https://www.loom.com/share/872b75c72b434dd586ff0086e791ff05"}

DOCUMENT15 = {
    "title": "Hey Daniel, \
            Thanks for reaching out. I tool sometime to go through https://relevanceai.com/ \
            Happy to get together for a chat may be over a coffee to discuss. \
            You can reach me at 650 422 9928 and we can play the sync up time and location. \
            Thanks, \
            Ajay",
    "content": "Hey Ajay, sounds good - I'll send you a text. Thanks, Daniel"}

DOCUMENT16 = {
    "title": "Hey Rosh, yes please send it to me.",
    "content": "Hi Christophe, \
              Here is a short video that shows the capabilities on a high level: \
              https://www.loom.com/share/872b75c72b434dd586ff0086e791ff05 \
              I'd be very interested in your thoughts and feedback. Could this platform provide value to you?"}

DOCUMENT17 = {
    "title": "Thanks, but we are currently not interested. Sorry for the negative reply. \
            Regards, Stephan",
    "content": "Hi Stephan, thank you for the reply, no worries at all! \
              Please feel free to get in touch with me whenever you think we could be of value to you."}

DOCUMENT18 = {
    "title": "Hi Rosh \
            Thank you for the invite. Currently, we aren't evsluating any vendor for the purpose of data authentication. \
            Best, \
            Stefan",
    "content": "Hi Stefan, thank you for the reply, I appreciate it. \
              That makes sense, please feel free to get in touch with me whenever you think we could provide value to you. \
              Have a great day, \
              Rosh"}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3, DOCUMENT4, DOCUMENT5, DOCUMENT6, DOCUMENT7, DOCUMENT8, DOCUMENT9, DOCUMENT10, DOCUMENT11, DOCUMENT12, DOCUMENT13, DOCUMENT14, DOCUMENT15, DOCUMENT16, DOCUMENT17, DOCUMENT18]

df = pd.DataFrame(documents)
df.columns = ['Title', 'Content']

# Get the embeddings of each text and add to an embeddings column in the dataframe
def embed_fn(title, text):
  return genai.embed_content(model=embedding_model,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]

df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Content']), axis=1)

def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model=embedding_model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['Content'] # Return text from index with max value

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""You are a world class business development representative. \
                           I will share a '{query}' message with you and you will give me the best answer that based on '{relevant_passage}'. \
                           Be sure to give respond message as this template. \
                           Hi [Prospect's Name], \
                           \
                           '{relevant_passage}' \
                           \
                           Best regards, [Your Name]\
                           \
                           Just make sure use this template, and then the content from the template using '{relevant_passage}' as a customer response. \
                           Follow ALL of the rules below of the response:\
                           1. Response should be very similar or even identical to the past best practices such as '{relevant_passage}',\
                           in terms of length, ton of voice, logical arguments and other details. \
                           """).format(query=query, relevant_passage=escaped)
  return prompt

# Streamlit app title
st.title("Customer Response Generator")

# Streamlit form for customer message input
with st.form(key='customer_message_form'):
    customer_message = st.text_area("Enter Customer Message:")
    submit_button = st.form_submit_button(label="Submit (Ctrl + Enter)")

if submit_button and customer_message:
    with st.spinner("Generating best practice message..."):
        passage = find_best_passage(customer_message, df)
        prompt = make_prompt(customer_message, passage)
        answer = model_generation.generate_content(prompt)
        
        # Calculate the height of the text area based on the length of the generated answer
        response_length = len(answer.text)
        height = max(200, min(800, response_length // 2))  # Adjust these values as needed

        st.text_area("Generated Response:", answer.text, height=height)