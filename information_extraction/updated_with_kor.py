import enum
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from kor import create_extraction_chain, Object, Text, Number
import pydantic
from typing import List
from kor import from_pydantic
from pydantic import BaseModel, Field
from typing import Optional
from config import set_environment

set_environment()

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
)

class Experience(BaseModel):
    # the title doesn't seem to help at all.
    start_date: Optional[str] = Field(description="When the job or study started.")
    end_date: Optional[str] = Field(description="When the job or study ended.")
    description: Optional[str] = Field(description="What the job or study entailed.")
    country: Optional[str] = Field(description="The country of the institution.")


class Study(Experience):
    degree: Optional[str] = Field(description="The degree obtained or expected.")
    institution: Optional[str] = Field(
        description="The university, college, or educational institution visited."
    )
    country: Optional[str] = Field(description="The country of the institution.")
    grade: Optional[str] = Field(description="The grade achieved or expected.")


class WorkExperience(Experience):
    company: str = Field(description="The company name of the work experience.")
    job_title: Optional[str] = Field(description="The job title.")


class Resume(BaseModel):
    first_name: Optional[str] = Field(description="The first name of the person.")
    last_name: Optional[str] = Field(description="The last name of the person.")
    linkedin_url: Optional[str] = Field(
        description="The url of the linkedin profile of the person."
    )
    email_address: Optional[str] = Field(description="The email address of the person.")
    nationality: Optional[str] = Field(description="The nationality of the person.")
    skill: Optional[str] = Field(description="A skill listed or mentioned in a description.")
    study: Optional[Study] = Field(
        description="A study that the person completed or is in progress of completing."
    )
    work_experience: Optional[WorkExperience] = Field(
        description="A work experience of the person."
    )
    hobby: Optional[str] = Field(description="A hobby or recreational activity of the person.")

#Extraer informacion del pdf
pdf_loader = PyPDFLoader("openresume-resume.pdf")
docs = pdf_loader.load_and_split()


model_json_schema, validator = from_pydantic(Resume)
chain = create_extraction_chain(
    llm, model_json_schema, encoder_or_encoder_class="json", validator=validator
)

print(chain.invoke(docs[0].page_content)["text"]["data"] )

