import pandas as pd

data = pd.read_csv('original_dataset.csv')

education_map = {
    1: "high school",
    2: "bachelors degree",
    3: "masters degree",
    4: "PhD"
}

gender_map = {
    0: "female",
    1: "male"
}

recruitment_map = {
    1: "campus recruitment",
    2: "job portal",
    3: "referral"
}

def generate_resume_text(row):
    pronoun = "she" if row['Gender'] == 0 else "he"  
    return (
        f"{gender_map[row['Gender']]} applicant, age {row['Age']}. {pronoun.capitalize()} has "
        f"{education_map[row['EducationLevel']]} and "
        f"{row['ExperienceYears']} years of experience. {pronoun.capitalize()} has "
        f"worked at {row['PreviousCompanies']} companies and "
        f"lives {row['DistanceFromCompany']:.1f} km from office. {pronoun.capitalize()} "
        f"scored {row['InterviewScore']}/100 in the interview, "
        f"{row['SkillScore']}/100 on skill tests, and "
        f"{row['PersonalityScore']}/100 on personality assessments. "
        f"{pronoun.capitalize()} applied via {recruitment_map[row['RecruitmentStrategy']]}."
    )

data['ResumeText'] = data.apply(generate_resume_text, axis=1)

output_data = data.copy()

output_data.to_csv('candidates_with_resumes.csv', index=False)



