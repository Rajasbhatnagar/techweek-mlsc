import os
import spacy
from flask import Flask, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from pyresparser import ResumeParser
import io
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter

app = Flask(__name__)
CORS(app, origins="http://localhost:5173")

# Ensure spaCy model is installed
try:
    spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading SpaCy model: {e}. Installing the model...")
    os.system("python -m spacy download en_core_web_sm")


class SkillExtractorML:
    def __init__(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise

        self.skills = {
            "frontend developer": ["JavaScript", "React.js", "HTML/CSS", "Bootstrap", "Tailwind CSS"],
            "backend developer": ["Node.js", "Express.js", "SQL", "NoSQL"],
            "app developer": ["Flutter", "Dart", "Swift", "Kotlin"],
            "devops engineer": ["Docker", "Kubernetes", "AWS", "Azure"],
            "website developer": ["HTML", "CSS", "JavaScript", "React.js", "Bootstrap"],
            "software developer": ["C/C++", "Java", "DSA", "OOPS", "DBMS"]
        }

        self.general_skills = ["Python", "JavaScript", "React.js", "HTML/CSS", "Node.js", "Express.js",
                               "SQL", "NoSQL", "Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
                               "Django", "Flask", "HTML", "CSS", "Flutter", "Dart"]

        self.skill_embeddings = self._get_skill_embeddings()
        self.general_skill_embeddings = self._get_general_skill_embeddings()

    def _embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def _get_skill_embeddings(self):
        skill_embeddings = {}
        for category, skills in self.skills.items():
            skill_embeddings[category] = np.vstack([self._embed_text(skill) for skill in skills])
        return skill_embeddings

    def _get_general_skill_embeddings(self):
        return np.vstack([self._embed_text(skill) for skill in self.general_skills])

    def extract_skills(self, job_title, resume_skills):
        matched_category = next((category for category in self.skills if category in job_title.lower()), None)
        relevant_skills = self.skills[matched_category] if matched_category else self._find_similar_skills(job_title)
        return [skill for skill in relevant_skills if skill not in resume_skills]

    def _find_similar_skills(self, job_title):
        job_title_embedding = self._embed_text(job_title)
        similarities = np.dot(job_title_embedding, self.general_skill_embeddings.T).flatten()
        threshold = 0.7
        return [skill for skill, sim in zip(self.general_skills, similarities) if sim > threshold]


extractor = SkillExtractorML()


def pdf_reader(file_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(file_path, 'rb') as file:
        for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()

    return text


def setup_pyresparser_config():
    config_path = os.path.expanduser("~/.pyresparser")
    os.makedirs(config_path, exist_ok=True)
    config_file = os.path.join(config_path, "config.cfg")
    
    # Create a valid config file if not already present
    if not os.path.exists(config_file):
        with open(config_file, 'w') as file:
            file.write(
                "[DEFAULT]\n"
                "DEBUG=False\n"
                "SKILLS_FILE=/path/to/skills.json\n"
                "NAME_REGEX=[A-Za-z]+ [A-Za-z]+\n"
                "EMAIL_REGEX=[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\n"
                "PHONE_REGEX=\+?[1-9][0-9]{7,14}\n"
                "LOCATION_REGEX=[A-Za-z]+(?: [A-Za-z]+)*\n"
            )
        print(f"Config file created at {config_file}. Please update 'SKILLS_FILE' with the correct path.")


def process_resume(file_path, job_title):
    try:
        if not os.path.exists(file_path):
            return {'error': f"File '{file_path}' not found."}

        # Read resume content
        resume_text = pdf_reader(file_path)

        # Parse the resume using pyresparser
        setup_pyresparser_config()  # Ensure config is set up
        resume_data = ResumeParser(file_path).get_extracted_data()
        extracted_skills = resume_data.get("skills", []) if resume_data else []

        if not job_title:
            return {'error': 'Job title is required'}

        recommended_skills = extractor.extract_skills(job_title, extracted_skills)
        return {
            'extracted_skills': extracted_skills,
            'recommended_skills': recommended_skills
        }
    except Exception as e:
        return {'error': str(e)}


@app.route('/process_resume/<job_title>/<path:file_path>', methods=['GET'])
def process_resume_endpoint(job_title, file_path):
    result = process_resume(file_path, job_title)
    return jsonify(result)


if __name__ == '__main__':
    # Test the function directly by providing file path and job title
    file_path = "resume-BJhEUnPK.pdf"  # Replace with the path to your test PDF file
    job_title = "frontend developer"  # Replace with your test job title

    if os.path.exists(file_path):
        result = process_resume(file_path, job_title)
        print("Processed Resume Output:")
        print(result)
    else:
        print(f"Error: File '{file_path}' not found.")
