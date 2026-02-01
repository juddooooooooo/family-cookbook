import streamlit as st
import json
import os
import time
try:
    from github import Github
except Exception:
    Github = None
import google.generativeai as genai
from PIL import Image
import io

# --- CONFIGURATION & SECRETS ---
st.set_page_config(page_title="Grandma's Recipes", page_icon="🥘", layout="wide")

# We get these from Streamlit Secrets (configured later on the cloud)
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_NAME = st.secrets["REPO_NAME"] # e.g., "juddjocum/family-cookbook"
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("Secrets not found! Please set GITHUB_TOKEN, REPO_NAME, and GEMINI_API_KEY in Streamlit Cloud.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# --- FUNCTIONS ---

def get_repo():
    """Connect to the GitHub Repo."""
    if Github is None:
        st.error("PyGithub is not installed. Please add 'PyGithub' to your environment (pip install PyGithub) and redeploy.")
        st.stop()
    g = Github(GITHUB_TOKEN)
    return g.get_repo(REPO_NAME)

@st.cache_data(ttl=60) # Refresh cache every 60 seconds
def load_data_from_github():
    """Fetch JSON and Images directly from GitHub."""
    repo = get_repo()
    try:
        # Get JSON content
        content = repo.get_contents("grandmother_recipes.json")
        json_data = json.loads(content.decoded_content.decode())
        return json_data
    except:
        return []

def save_to_github(new_recipe, image_file, image_name):
    """Push the new image and updated JSON back to GitHub."""
    repo = get_repo()
    
    # 1. Upload the Image
    try:
        repo.create_file(
            path=f"Recipe_images/{image_name}",
            message=f"Add image: {image_name}",
            content=image_file.getvalue()
        )
    except Exception as e:
        st.error(f"Error uploading image: {e}")
        return False

    # 2. Update the JSON
    try:
        # Get current file to find the 'sha' (required for updating)
        contents = repo.get_contents("grandmother_recipes.json")
        current_data = json.loads(contents.decoded_content.decode())
        
        # Append new recipe
        current_data.append(new_recipe)
        
        # Update file
        repo.update_file(
            path=contents.path,
            message=f"Add recipe: {new_recipe['title']}",
            content=json.dumps(current_data, indent=4, ensure_ascii=False),
            sha=contents.sha
        )
        return True
    except Exception as e:
        st.error(f"Error updating JSON: {e}")
        return False

def extract_recipe_gemini(uploaded_file):
    """Send image to Gemini for extraction."""
    with st.spinner("🤖 Analyzing recipe with AI... (this takes ~5 seconds)"):
        try:
            # Create a temporary image for Gemini
            img = Image.open(uploaded_file)
            
            prompt = """
            Extract the recipe from this image.
            Return a valid JSON object with this EXACT structure:
            {
                "title": "Recipe Name",
                "author": "Name of person mentioned (or null)",
                "category": "Choose one: [Starter, Main, Dessert, Baking, Breakfast, Side, Drinks]",
                "ingredients": ["item 1", "item 2"],
                "instructions": ["step 1", "step 2"],
                "notes": "Any handwritten notes or context",
                "servings": "Servings (or null)"
            }
            """
            response = model.generate_content([prompt, img])
            text = response.text.strip()
            # Clean Markdown
            if text.startswith("```json"): text = text[7:-3]
            elif text.startswith("```"): text = text[3:-3]
            return json.loads(text)
        except Exception as e:
            st.error(f"AI Extraction failed: {e}")
            return None

# --- UI LAYOUT ---

# Define tabs
tab1, tab2 = st.tabs(["📖 Cookbook", "📤 Upload New Recipe"])

# === TAB 1: VIEW RECIPES ===
with tab1:
    recipes = load_data_from_github()
    
    # Sidebar Filters (Moved here so they only appear on View tab)
    st.sidebar.header("Filter Recipes")
    categories = sorted(list(set([r.get('category', 'Other') for r in recipes]))) if recipes else []
    cat_filter = st.sidebar.multiselect("Category", categories)
    search = st.sidebar.text_input("Search", "")

    # Filter Logic
    filtered = recipes
    if cat_filter:
        filtered = [r for r in filtered if r.get('category') in cat_filter]
    if search:
        term = search.lower()
        filtered = [r for r in filtered if term in r['title'].lower() or any(term in i.lower() for i in r['ingredients'])]

    if not filtered:
        st.info("No recipes found.")
    else:
        # Display Grid
        cols = st.columns(2)
        for idx, r in enumerate(filtered):
            with cols[idx % 2]:
                with st.container(border=True):
                    # Image from GitHub Raw URL
                    if r.get('source_file'):
                        # Construct raw github url for display
                        # Format: https://raw.githubusercontent.com/{USER}/{REPO}/main/Recipe_images/{FILE}
                        img_url = f"https://raw.githubusercontent.com/{REPO_NAME}/main/Recipe_images/{r['source_file']}"
                        st.image(img_url, use_container_width=True)
                    
                    st.subheader(r['title'])
                    st.caption(f"{r.get('category', 'General')} | By {r.get('author', 'Unknown')}")
                    
                    with st.expander("Ingredients & Steps"):
                        st.write(f"_{r.get('notes', '')}_")
                        st.markdown("### Ingredients")
                        for i in r['ingredients']: st.markdown(f"- {i}")
                        st.markdown("### Instructions")
                        for n, step in enumerate(r['instructions']): st.markdown(f"{n+1}. {step}")

# === TAB 2: UPLOAD ===
with tab2:
    st.header("Upload a Family Recipe")
    st.write("Take a photo of a recipe card, upload it here, and AI will add it to the book!")
    
    uploaded_file = st.file_uploader("Choose a photo...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # 1. Preview
        st.image(uploaded_file, caption="Preview", width=300)
        
        # 2. Extract Button
        if st.button("✨ Analyze & Add Recipe"):
            extracted_data = extract_recipe_gemini(uploaded_file)
            
            if extracted_data:
                # Show Editable Form (in case AI made a mistake)
                st.success("AI extraction complete! Please verify details below:")
                
                with st.form("confirm_upload"):
                    new_title = st.text_input("Title", extracted_data['title'])
                    new_author = st.text_input("Author (Grandma, Auntie...)", extracted_data.get('author', ''))
                    new_cat = st.selectbox("Category", ["Starter", "Main", "Dessert", "Baking", "Breakfast", "Side", "Drinks"], index=2)
                    
                    # Store raw lists as text for editing
                    ing_text = st.text_area("Ingredients (one per line)", "\n".join(extracted_data['ingredients']))
                    inst_text = st.text_area("Instructions (one per line)", "\n".join(extracted_data['instructions']))
                    
                    submitted = st.form_submit_button("💾 Save to Family Cookbook")
                    
                    if submitted:
                        # Reconstruct the object
                        final_recipe = {
                            "title": new_title,
                            "author": new_author,
                            "category": new_cat,
                            "ingredients": ing_text.split('\n'),
                            "instructions": inst_text.split('\n'),
                            "notes": extracted_data.get('notes', ''),
                            "servings": extracted_data.get('servings'),
                            "source_file": uploaded_file.name, # We will use the original filename
                            "date_added": time.strftime("%Y-%m-%d")
                        }
                        
                        # Save to GitHub
                        with st.spinner("Saving to database..."):
                            success = save_to_github(final_recipe, uploaded_file, uploaded_file.name)
                            if success:
                                st.balloons()
                                st.success("Recipe Saved! Go to the 'Cookbook' tab to see it.")
                                st.cache_data.clear() # Clear cache so new recipe shows up