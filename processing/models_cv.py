from pydantic import BaseModel
from typing import Optional
from enum import Enum

# Enums for standardized options
class ProficiencyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    NATIVE = "native" 

class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"
    VOLUNTEER = "volunteer"

class ProjectType(str, Enum):
    PERSONAL = "personal"
    ACADEMIC = "academic"
    PROFESSIONAL = "professional"
    FREELANCE = "freelance"
    OPEN_SOURCE = "open_source"
    VOLUNTEER = "volunteer"

class Seniority(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"


class ResponseFormatter(BaseModel):
    contact_information_full_name: Optional[str]
    contact_information_phone_number: Optional[str]
    contact_information_email: Optional[str]
    contact_information_linkedin: Optional[str]
    contact_information_address: Optional[str]
    personal_summary: Optional[str]
    education_degrees: Optional[str]
    education_field_of_study: Optional[str]
    education_institutions: Optional[str]
    education_locations: Optional[str]
    education_honors: Optional[str]
    education_descriptions: Optional[str]
    work_experience_job_titles: Optional[str]
    work_experience_employers: Optional[str]
    work_experience_industry: Optional[str]
    work_experience_employment_type: Optional[EmploymentType]
    work_experience_locations: Optional[str]
    work_experience_descriptions: Optional[str]
    work_experience_seniority: Optional[Seniority]
    skills: Optional[str]
    project_titles: Optional[str]
    project_project_types: Optional[str]
    project_descriptions: Optional[str]
    project_roles: Optional[str]
    project_tools_technologies: Optional[str]
    certification_names: Optional[str]
    certification_descriptions: Optional[str]
    certification_issuing_organizations: Optional[str]
    publication_titles: Optional[str]
    publication_publishers: Optional[str]
    publication_descriptions: Optional[str]
    language_languages: Optional[str]
    language_proficiencies: Optional[str]
    award_and_honor_titles: Optional[str]
    award_and_honor_issuing_organizations: Optional[str]
    award_and_honor_descriptions: Optional[str]
    volunteer_experience_roles: Optional[str]
    volunteer_experience_organizations: Optional[str]
    volunteer_experience_descriptions: Optional[str]
