"""
Human Evaluation Framework for Explainable Face Recognition
==========================================================

Comprehensive system for conducting human studies to evaluate:
- User understanding of AI explanations
- Trust in face recognition decisions  
- Explanation plausibility and usefulness
- Demographic bias in perception
"""

import os
import sys
import json
import csv
import uuid
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import random
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class ParticipantDemographics:
    """Participant demographic information"""
    participant_id: str
    age_group: str  # "18-25", "26-35", "36-45", "46-55", "56+"
    gender: str
    education_level: str  # "high_school", "bachelor", "master", "doctorate", "other"
    tech_experience: str  # "low", "medium", "high"
    ai_familiarity: str  # "none", "basic", "intermediate", "advanced"
    face_recognition_experience: str  # "none", "basic", "experienced"
    occupation_category: str
    consent_timestamp: str
    study_session_id: str

@dataclass
class ExplanationRating:
    """Individual explanation rating by participant"""
    participant_id: str
    explanation_id: str
    explanation_type: str  # "visual", "textual", "attribute", "prototype", "counterfactual"
    
    # Understanding metrics (1-7 scale)
    clarity_rating: int  # How clear was the explanation?
    completeness_rating: int  # How complete was the explanation?
    technical_understanding: int  # How well did you understand the technical aspects?
    
    # Trust metrics (1-7 scale)
    trust_in_decision: int  # How much do you trust this AI decision?
    trust_in_explanation: int  # How much do you trust this explanation?
    confidence_in_system: int  # How confident are you in this system?
    
    # Utility metrics (1-7 scale)
    usefulness_rating: int  # How useful was this explanation?
    actionability_rating: int  # How actionable was this information?
    plausibility_rating: int  # How plausible was the explanation?
    
    # Additional feedback
    most_helpful_aspect: str
    least_helpful_aspect: str
    improvement_suggestions: str
    free_text_feedback: str
    
    # Timing data
    time_to_decision: float  # seconds
    time_viewing_explanation: float  # seconds
    total_time: float  # seconds
    
    # System data
    timestamp: str
    session_id: str

@dataclass
class TaskResult:
    """Results from a specific task in the study"""
    participant_id: str
    task_id: str
    task_type: str  # "verification", "identification", "explanation_rating"
    
    # Task-specific data
    correct_answer: bool
    participant_answer: bool
    confidence_in_answer: int  # 1-7 scale
    
    # Explanation data
    explanation_viewed: bool
    explanation_influenced_decision: bool  # Did explanation change your mind?
    pre_explanation_confidence: int
    post_explanation_confidence: int
    
    # Performance metrics
    accuracy: float
    response_time: float
    number_of_interactions: int
    
    # Metadata
    stimulus_id: str
    ground_truth_label: str
    ai_prediction: str
    ai_confidence: float
    
    timestamp: str

class HumanEvaluationFramework:
    """Main framework for conducting human evaluation studies"""
    
    def __init__(self, study_name: str, output_dir: str = "human_studies"):
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Study configuration
        self.target_participants = 100  # N=30-100 as requested
        self.min_participants = 30
        self.tasks_per_participant = 20
        
        # Data storage
        self.participants: Dict[str, ParticipantDemographics] = {}
        self.ratings: List[ExplanationRating] = []
        self.task_results: List[TaskResult] = []
        
        # Study materials
        self.stimulus_set = self._prepare_stimulus_set()
        self.explanation_templates = self._load_explanation_templates()
        
        self._initialize_study()
    
    def _initialize_study(self):
        """Initialize study files and directories"""
        # Create directory structure
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "consent_forms").mkdir(exist_ok=True)
        (self.output_dir / "stimuli").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Initialize data files
        self._initialize_data_files()
        
        print(f"✅ Human evaluation study '{self.study_name}' initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Target participants: {self.target_participants}")
    
    def _initialize_data_files(self):
        """Initialize CSV files for data collection"""
        # Participants file
        participants_file = self.output_dir / "data" / "participants.csv"
        if not participants_file.exists():
            with open(participants_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'participant_id', 'age_group', 'gender', 'education_level',
                    'tech_experience', 'ai_familiarity', 'face_recognition_experience',
                    'occupation_category', 'consent_timestamp', 'study_session_id'
                ])
        
        # Ratings file
        ratings_file = self.output_dir / "data" / "explanation_ratings.csv"
        if not ratings_file.exists():
            with open(ratings_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'participant_id', 'explanation_id', 'explanation_type',
                    'clarity_rating', 'completeness_rating', 'technical_understanding',
                    'trust_in_decision', 'trust_in_explanation', 'confidence_in_system',
                    'usefulness_rating', 'actionability_rating', 'plausibility_rating',
                    'most_helpful_aspect', 'least_helpful_aspect', 'improvement_suggestions',
                    'free_text_feedback', 'time_to_decision', 'time_viewing_explanation',
                    'total_time', 'timestamp', 'session_id'
                ])
        
        # Task results file
        tasks_file = self.output_dir / "data" / "task_results.csv"
        if not tasks_file.exists():
            with open(tasks_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'participant_id', 'task_id', 'task_type', 'correct_answer',
                    'participant_answer', 'confidence_in_answer', 'explanation_viewed',
                    'explanation_influenced_decision', 'pre_explanation_confidence',
                    'post_explanation_confidence', 'accuracy', 'response_time',
                    'number_of_interactions', 'stimulus_id', 'ground_truth_label',
                    'ai_prediction', 'ai_confidence', 'timestamp'
                ])
    
    def _prepare_stimulus_set(self):
        """Prepare standardized stimulus set for evaluation"""
        # In a real study, this would load actual face pairs with known ground truth
        # For framework demonstration, we'll create mock stimuli
        
        stimulus_categories = [
            "easy_match",      # Clear match cases
            "easy_non_match",  # Clear non-match cases
            "difficult_match", # Ambiguous match cases
            "difficult_non_match", # Ambiguous non-match cases
            "demographic_bias_test", # Cases testing for bias
            "edge_cases"       # Unusual cases (glasses, lighting, etc.)
        ]
        
        stimuli = []
        for category in stimulus_categories:
            for i in range(20):  # 20 stimuli per category
                stimulus = {
                    'id': f"{category}_{i:03d}",
                    'category': category,
                    'image1_path': f"stimuli/{category}/image1_{i:03d}.jpg",
                    'image2_path': f"stimuli/{category}/image2_{i:03d}.jpg",
                    'ground_truth': random.choice([True, False]),
                    'difficulty_level': self._assign_difficulty(category),
                    'demographic_info': self._assign_demographics(category),
                    'ai_prediction': random.uniform(0.3, 0.9),
                    'explanation_data': self._generate_explanation_data(category, i)
                }
                stimuli.append(stimulus)
        
        return stimuli
    
    def _assign_difficulty(self, category: str) -> str:
        """Assign difficulty level based on category"""
        difficulty_map = {
            "easy_match": "easy",
            "easy_non_match": "easy", 
            "difficult_match": "hard",
            "difficult_non_match": "hard",
            "demographic_bias_test": "medium",
            "edge_cases": "hard"
        }
        return difficulty_map.get(category, "medium")
    
    def _assign_demographics(self, category: str) -> Dict[str, str]:
        """Assign demographic information for bias testing"""
        if category == "demographic_bias_test":
            return {
                'apparent_race': random.choice(['white', 'black', 'asian', 'hispanic', 'other']),
                'apparent_gender': random.choice(['male', 'female']),
                'apparent_age': random.choice(['young', 'middle', 'old'])
            }
        else:
            return {
                'apparent_race': 'mixed',
                'apparent_gender': 'mixed', 
                'apparent_age': 'mixed'
            }
    
    def _generate_explanation_data(self, category: str, index: int) -> Dict[str, Any]:
        """Generate explanation data for stimulus"""
        return {
            'visual_explanation': f"visual_explanation_{category}_{index}.png",
            'textual_explanation': f"Explanation for {category} case {index}",
            'attributes': self._generate_mock_attributes(),
            'prototypes': self._generate_mock_prototypes(),
            'counterfactuals': self._generate_mock_counterfactuals()
        }
    
    def _generate_mock_attributes(self) -> List[Dict[str, float]]:
        """Generate mock attribute analysis"""
        attributes = ['Male', 'Young', 'Smiling', 'Eyeglasses', 'Beard', 'Hat']
        return [
            {'name': attr, 'confidence': random.uniform(-1, 1)}
            for attr in attributes
        ]
    
    def _generate_mock_prototypes(self) -> List[Dict[str, Any]]:
        """Generate mock prototype data"""
        return [
            {
                'id': i,
                'similarity': random.uniform(0.7, 0.95),
                'description': f'Prototype {i} description'
            }
            for i in range(3)
        ]
    
    def _generate_mock_counterfactuals(self) -> List[Dict[str, Any]]:
        """Generate mock counterfactual data"""
        return [
            {
                'modification': 'Add glasses',
                'impact': random.uniform(-0.1, 0.1),
                'confidence': random.uniform(0.8, 0.95)
            },
            {
                'modification': 'Change expression',
                'impact': random.uniform(-0.05, 0.05),
                'confidence': random.uniform(0.8, 0.95)
            }
        ]
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates for different types"""
        return {
            'visual': "Visual explanation showing attention maps and key facial regions",
            'textual': "Text-based explanation describing the AI's reasoning process",
            'attribute': "Attribute-based explanation listing facial characteristics",
            'prototype': "Prototype-based explanation showing similar faces",
            'counterfactual': "Counterfactual explanation showing what-if scenarios"
        }
    
    def register_participant(self, demographics_data: Dict[str, str]) -> str:
        """Register a new participant and return their ID"""
        participant_id = f"P{len(self.participants) + 1:03d}_{uuid.uuid4().hex[:8]}"
        
        participant = ParticipantDemographics(
            participant_id=participant_id,
            age_group=demographics_data.get('age_group', 'unknown'),
            gender=demographics_data.get('gender', 'unknown'),
            education_level=demographics_data.get('education_level', 'unknown'),
            tech_experience=demographics_data.get('tech_experience', 'unknown'),
            ai_familiarity=demographics_data.get('ai_familiarity', 'unknown'),
            face_recognition_experience=demographics_data.get('face_recognition_experience', 'unknown'),
            occupation_category=demographics_data.get('occupation_category', 'unknown'),
            consent_timestamp=datetime.datetime.now().isoformat(),
            study_session_id=uuid.uuid4().hex
        )
        
        self.participants[participant_id] = participant
        self._save_participant_data(participant)
        
        print(f"✅ Participant registered: {participant_id}")
        print(f"📊 Total participants: {len(self.participants)}")
        
        return participant_id
    
    def _save_participant_data(self, participant: ParticipantDemographics):
        """Save participant data to CSV"""
        participants_file = self.output_dir / "data" / "participants.csv"
        
        with open(participants_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                participant.participant_id,
                participant.age_group,
                participant.gender,
                participant.education_level,
                participant.tech_experience,
                participant.ai_familiarity,
                participant.face_recognition_experience,
                participant.occupation_category,
                participant.consent_timestamp,
                participant.study_session_id
            ])
    
    def get_participant_tasks(self, participant_id: str) -> List[Dict[str, Any]]:
        """Get randomized task set for a participant"""
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")
        
        # Stratified sampling to ensure representation across categories
        tasks = []
        categories = list(set(s['category'] for s in self.stimulus_set))
        
        # Ensure each participant sees examples from each category
        tasks_per_category = self.tasks_per_participant // len(categories)
        remainder = self.tasks_per_participant % len(categories)
        
        for i, category in enumerate(categories):
            category_stimuli = [s for s in self.stimulus_set if s['category'] == category]
            
            # Add extra task to some categories if remainder exists
            num_tasks = tasks_per_category + (1 if i < remainder else 0)
            
            selected_stimuli = random.sample(category_stimuli, min(num_tasks, len(category_stimuli)))
            
            for stimulus in selected_stimuli:
                task = {
                    'task_id': f"{participant_id}_task_{len(tasks):03d}",
                    'stimulus': stimulus,
                    'explanation_type': random.choice(['visual', 'textual', 'attribute', 'prototype', 'counterfactual']),
                    'show_explanation': random.choice([True, False])  # A/B testing
                }
                tasks.append(task)
        
        # Randomize task order
        random.shuffle(tasks)
        
        return tasks
    
    def submit_rating(self, rating_data: Dict[str, Any]) -> str:
        """Submit an explanation rating"""
        rating = ExplanationRating(
            participant_id=rating_data['participant_id'],
            explanation_id=rating_data['explanation_id'],
            explanation_type=rating_data['explanation_type'],
            clarity_rating=rating_data['clarity_rating'],
            completeness_rating=rating_data['completeness_rating'],
            technical_understanding=rating_data['technical_understanding'],
            trust_in_decision=rating_data['trust_in_decision'],
            trust_in_explanation=rating_data['trust_in_explanation'],
            confidence_in_system=rating_data['confidence_in_system'],
            usefulness_rating=rating_data['usefulness_rating'],
            actionability_rating=rating_data['actionability_rating'],
            plausibility_rating=rating_data['plausibility_rating'],
            most_helpful_aspect=rating_data.get('most_helpful_aspect', ''),
            least_helpful_aspect=rating_data.get('least_helpful_aspect', ''),
            improvement_suggestions=rating_data.get('improvement_suggestions', ''),
            free_text_feedback=rating_data.get('free_text_feedback', ''),
            time_to_decision=rating_data.get('time_to_decision', 0.0),
            time_viewing_explanation=rating_data.get('time_viewing_explanation', 0.0),
            total_time=rating_data.get('total_time', 0.0),
            timestamp=datetime.datetime.now().isoformat(),
            session_id=rating_data.get('session_id', '')
        )
        
        self.ratings.append(rating)
        self._save_rating_data(rating)
        
        rating_id = f"R{len(self.ratings):06d}"
        print(f"✅ Rating submitted: {rating_id}")
        
        return rating_id
    
    def _save_rating_data(self, rating: ExplanationRating):
        """Save rating data to CSV"""
        ratings_file = self.output_dir / "data" / "explanation_ratings.csv"
        
        with open(ratings_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                rating.participant_id,
                rating.explanation_id,
                rating.explanation_type,
                rating.clarity_rating,
                rating.completeness_rating,
                rating.technical_understanding,
                rating.trust_in_decision,
                rating.trust_in_explanation,
                rating.confidence_in_system,
                rating.usefulness_rating,
                rating.actionability_rating,
                rating.plausibility_rating,
                rating.most_helpful_aspect,
                rating.least_helpful_aspect,
                rating.improvement_suggestions,
                rating.free_text_feedback,
                rating.time_to_decision,
                rating.time_viewing_explanation,
                rating.total_time,
                rating.timestamp,
                rating.session_id
            ])
    
    def submit_task_result(self, result_data: Dict[str, Any]) -> str:
        """Submit a task result"""
        result = TaskResult(
            participant_id=result_data['participant_id'],
            task_id=result_data['task_id'],
            task_type=result_data['task_type'],
            correct_answer=result_data['correct_answer'],
            participant_answer=result_data['participant_answer'],
            confidence_in_answer=result_data['confidence_in_answer'],
            explanation_viewed=result_data.get('explanation_viewed', False),
            explanation_influenced_decision=result_data.get('explanation_influenced_decision', False),
            pre_explanation_confidence=result_data.get('pre_explanation_confidence', 0),
            post_explanation_confidence=result_data.get('post_explanation_confidence', 0),
            accuracy=1.0 if result_data['correct_answer'] == result_data['participant_answer'] else 0.0,
            response_time=result_data.get('response_time', 0.0),
            number_of_interactions=result_data.get('number_of_interactions', 1),
            stimulus_id=result_data['stimulus_id'],
            ground_truth_label=result_data['ground_truth_label'],
            ai_prediction=result_data['ai_prediction'],
            ai_confidence=result_data['ai_confidence'],
            timestamp=datetime.datetime.now().isoformat()
        )
        
        self.task_results.append(result)
        self._save_task_result(result)
        
        result_id = f"T{len(self.task_results):06d}"
        print(f"✅ Task result submitted: {result_id}")
        
        return result_id
    
    def _save_task_result(self, result: TaskResult):
        """Save task result to CSV"""
        tasks_file = self.output_dir / "data" / "task_results.csv"
        
        with open(tasks_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.participant_id,
                result.task_id,
                result.task_type,
                result.correct_answer,
                result.participant_answer,
                result.confidence_in_answer,
                result.explanation_viewed,
                result.explanation_influenced_decision,
                result.pre_explanation_confidence,
                result.post_explanation_confidence,
                result.accuracy,
                result.response_time,
                result.number_of_interactions,
                result.stimulus_id,
                result.ground_truth_label,
                result.ai_prediction,
                result.ai_confidence,
                result.timestamp
            ])
    
    def get_study_status(self) -> Dict[str, Any]:
        """Get current study status and progress"""
        total_possible_ratings = len(self.participants) * self.tasks_per_participant
        completion_rate = len(self.ratings) / max(total_possible_ratings, 1) * 100
        
        # Demographic distribution
        demographics = {}
        for participant in self.participants.values():
            for key in ['age_group', 'gender', 'education_level', 'tech_experience', 'ai_familiarity']:
                if key not in demographics:
                    demographics[key] = {}
                value = getattr(participant, key)
                demographics[key][value] = demographics[key].get(value, 0) + 1
        
        # Rating statistics
        rating_stats = {}
        if self.ratings:
            for metric in ['clarity_rating', 'trust_in_decision', 'usefulness_rating']:
                values = [getattr(r, metric) for r in self.ratings]
                rating_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return {
            'study_name': self.study_name,
            'participants_recruited': len(self.participants),
            'target_participants': self.target_participants,
            'ratings_collected': len(self.ratings),
            'tasks_completed': len(self.task_results),
            'completion_rate': completion_rate,
            'demographics': demographics,
            'rating_statistics': rating_stats,
            'ready_for_analysis': len(self.participants) >= self.min_participants
        }
    
    def export_study_data(self) -> str:
        """Export all study data for analysis"""
        export_dir = self.output_dir / "exports" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export participants
        participants_export = export_dir / "participants.json"
        with open(participants_export, 'w') as f:
            json.dump([asdict(p) for p in self.participants.values()], f, indent=2)
        
        # Export ratings
        ratings_export = export_dir / "ratings.json"
        with open(ratings_export, 'w') as f:
            json.dump([asdict(r) for r in self.ratings], f, indent=2)
        
        # Export task results
        results_export = export_dir / "task_results.json"
        with open(results_export, 'w') as f:
            json.dump([asdict(r) for r in self.task_results], f, indent=2)
        
        # Export study metadata
        metadata_export = export_dir / "study_metadata.json"
        with open(metadata_export, 'w') as f:
            json.dump({
                'study_name': self.study_name,
                'export_timestamp': datetime.datetime.now().isoformat(),
                'stimulus_set_size': len(self.stimulus_set),
                'participants_count': len(self.participants),
                'ratings_count': len(self.ratings),
                'tasks_count': len(self.task_results)
            }, f, indent=2)
        
        print(f"✅ Study data exported to: {export_dir}")
        return str(export_dir)


class ParticipantRecruitmentManager:
    """Manager for participant recruitment and screening"""
    
    def __init__(self, target_demographics: Dict[str, Dict[str, int]]):
        self.target_demographics = target_demographics
        self.recruitment_channels = [
            'university_posting',
            'social_media',
            'research_database', 
            'referrals',
            'crowdsourcing_platform'
        ]
        
    def check_demographic_needs(self, current_participants: Dict[str, ParticipantDemographics]) -> Dict[str, int]:
        """Check which demographic groups need more participants"""
        current_counts = {}
        
        # Count current participants by demographic categories
        for participant in current_participants.values():
            key = f"{participant.age_group}_{participant.gender}"
            current_counts[key] = current_counts.get(key, 0) + 1
        
        # Calculate needed participants
        needs = {}
        for demo_key, target_count in self.target_demographics.items():
            current = current_counts.get(demo_key, 0)
            if current < target_count:
                needs[demo_key] = target_count - current
        
        return needs
    
    def generate_recruitment_materials(self) -> Dict[str, str]:
        """Generate recruitment materials"""
        return {
            'flyer_text': """
            🔬 RESEARCH STUDY: Understanding AI Explanations
            
            We are conducting a study on how people understand AI explanations 
            in face recognition systems. Your participation will help improve 
            AI transparency and trustworthiness.
            
            ⏰ Time: 45-60 minutes
            💰 Compensation: $15 gift card
            📍 Location: Online study
            
            Requirements:
            - Age 18 or older
            - Normal or corrected vision
            - English proficiency
            
            Contact: [research_email]
            """,
            
            'social_media_post': """
            Help us make AI more explainable! 🤖✨
            
            Join our research study on AI face recognition explanations. 
            45 minutes, $15 compensation, conducted online.
            
            Your input will directly improve AI systems!
            
            Sign up: [study_link]
            #AIResearch #HumanStudy #ExplainableAI
            """,
            
            'email_template': """
            Subject: Research Participation Opportunity - AI Explanation Study
            
            Dear [Name],
            
            You are invited to participate in a research study investigating 
            how people understand explanations from AI face recognition systems.
            
            This study will help us design better, more trustworthy AI systems.
            
            Study Details:
            - Duration: 45-60 minutes
            - Compensation: $15 Amazon gift card
            - Format: Online via secure web interface
            - Anonymous and confidential
            
            To participate, please visit: [study_link]
            
            Thank you for your time and contribution to AI research!
            
            Best regards,
            [Research Team]
            """
        }


def create_example_study():
    """Create an example human evaluation study"""
    
    # Initialize framework
    framework = HumanEvaluationFramework(
        study_name="explainable_face_recognition_study_2024",
        output_dir="human_studies"
    )
    
    # Example participant registration
    participant_data = {
        'age_group': '26-35',
        'gender': 'female',
        'education_level': 'master',
        'tech_experience': 'high',
        'ai_familiarity': 'intermediate',
        'face_recognition_experience': 'basic',
        'occupation_category': 'software_engineer'
    }
    
    participant_id = framework.register_participant(participant_data)
    
    # Example rating submission
    rating_data = {
        'participant_id': participant_id,
        'explanation_id': 'exp_001',
        'explanation_type': 'visual',
        'clarity_rating': 6,
        'completeness_rating': 5,
        'technical_understanding': 4,
        'trust_in_decision': 5,
        'trust_in_explanation': 6,
        'confidence_in_system': 5,
        'usefulness_rating': 7,
        'actionability_rating': 4,
        'plausibility_rating': 6,
        'most_helpful_aspect': 'Visual attention maps were clear',
        'least_helpful_aspect': 'Could use more technical detail',
        'improvement_suggestions': 'Add confidence intervals',
        'free_text_feedback': 'Overall very helpful explanation',
        'time_to_decision': 12.5,
        'time_viewing_explanation': 25.3,
        'total_time': 37.8,
        'session_id': 'session_001'
    }
    
    framework.submit_rating(rating_data)
    
    # Get study status
    status = framework.get_study_status()
    print("\n📊 Study Status:")
    print(json.dumps(status, indent=2))
    
    return framework

if __name__ == "__main__":
    print("🚀 Initializing Human Evaluation Framework...")
    
    # Create example study
    study = create_example_study()
    
    print("\n✅ Example study created successfully!")
    print(f"📁 Study files located in: {study.output_dir}")
    print(f"🎯 Framework ready for {study.target_participants} participants")