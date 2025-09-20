"""
Interactive Explanation Components
=================================

Enhanced interactive features for the dashboard including prototypes,
counterfactuals, and dynamic model exploration.
"""

class InteractiveExplanationHandler:
    """Handler for interactive explanation features"""
    
    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer
        self.prototype_cache = {}
        self.counterfactual_cache = {}
    
    def get_similar_prototypes(self, embedding, top_k=5):
        """Get most similar prototype faces"""
        # In a real implementation, this would query a prototype database
        # For demo, we'll return mock prototypes
        prototypes = [
            {
                'id': 1,
                'similarity': 0.89,
                'attributes': ['Male', 'Young', 'Brown_Hair', 'No_Beard'],
                'confidence': 0.95,
                'image_path': '/static/prototypes/proto_1.jpg',
                'description': 'Similar facial structure and age group'
            },
            {
                'id': 2,
                'similarity': 0.85,
                'attributes': ['Male', 'Smiling', 'No_Beard', 'Attractive'],
                'confidence': 0.92,
                'image_path': '/static/prototypes/proto_2.jpg',
                'description': 'Matching expression and facial features'
            },
            {
                'id': 3,
                'similarity': 0.82,
                'attributes': ['Male', 'Young', 'Attractive', 'Straight_Hair'],
                'confidence': 0.88,
                'image_path': '/static/prototypes/proto_3.jpg',
                'description': 'Similar age and hair characteristics'
            },
            {
                'id': 4,
                'similarity': 0.79,
                'attributes': ['Male', 'High_Cheekbones', 'Oval_Face'],
                'confidence': 0.84,
                'image_path': '/static/prototypes/proto_4.jpg',
                'description': 'Matching bone structure and face shape'
            },
            {
                'id': 5,
                'similarity': 0.76,
                'attributes': ['Male', 'Young', 'No_Eyeglasses', 'Clear_Skin'],
                'confidence': 0.81,
                'image_path': '/static/prototypes/proto_5.jpg',
                'description': 'Similar skin texture and clarity'
            }
        ]
        
        return prototypes[:top_k]
    
    def generate_counterfactual_glasses(self, image, current_similarity):
        """Generate counterfactual with glasses added/removed"""
        # Simulate the effect of adding/removing glasses
        has_glasses = self._detect_glasses(image)
        
        if has_glasses:
            # Simulate removing glasses
            new_similarity = current_similarity + 0.023  # Glasses typically hurt recognition
            modification = "Glasses Removed"
            effect = "Improved recognition accuracy"
            confidence = 0.91
        else:
            # Simulate adding glasses
            new_similarity = current_similarity - 0.054
            modification = "Glasses Added"
            effect = "Reduced recognition accuracy"
            confidence = 0.89
        
        return {
            'modification': modification,
            'original_similarity': current_similarity,
            'new_similarity': max(0, min(1, new_similarity)),
            'change': new_similarity - current_similarity,
            'effect': effect,
            'confidence': confidence,
            'explanation': f"{modification.lower()} typically {'improves' if new_similarity > current_similarity else 'reduces'} face recognition performance due to {'reduced occlusion' if has_glasses else 'facial occlusion'}."
        }
    
    def generate_counterfactual_expression(self, image, current_similarity):
        """Generate counterfactual with expression change"""
        current_expression = self._detect_expression(image)
        
        if current_expression == 'neutral':
            # Add smile
            new_similarity = current_similarity + 0.037
            modification = "Smile Added"
            new_attributes = ['Smiling', 'Mouth_Slightly_Open']
        elif current_expression == 'smiling':
            # Remove smile
            new_similarity = current_similarity - 0.025
            modification = "Smile Removed"
            new_attributes = ['Neutral_Expression', 'Closed_Mouth']
        else:
            # Change to neutral
            new_similarity = current_similarity + 0.012
            modification = "Expression Neutralized"
            new_attributes = ['Neutral_Expression']
        
        return {
            'modification': modification,
            'original_similarity': current_similarity,
            'new_similarity': max(0, min(1, new_similarity)),
            'change': new_similarity - current_similarity,
            'new_attributes': new_attributes,
            'confidence': 0.87,
            'explanation': f"Expression changes can {'improve' if new_similarity > current_similarity else 'reduce'} recognition by altering facial muscle configuration and feature visibility."
        }
    
    def generate_counterfactual_hair(self, image, current_similarity):
        """Generate counterfactual with hair style change"""
        # Simulate hair style modifications
        hair_modifications = [
            {
                'type': 'Hair Length Change',
                'change': -0.012,
                'description': 'Longer hair partially occludes facial features'
            },
            {
                'type': 'Hair Color Change',
                'change': 0.008,
                'description': 'Hair color change affects overall facial contrast'
            },
            {
                'type': 'Hair Style Change',
                'change': -0.005,
                'description': 'Different hair style slightly alters face shape perception'
            }
        ]
        
        selected_mod = hair_modifications[0]  # For demo, select first
        new_similarity = current_similarity + selected_mod['change']
        
        return {
            'modification': selected_mod['type'],
            'original_similarity': current_similarity,
            'new_similarity': max(0, min(1, new_similarity)),
            'change': selected_mod['change'],
            'confidence': 0.78,
            'explanation': selected_mod['description']
        }
    
    def get_sensitivity_analysis(self, image, current_similarity):
        """Get comprehensive sensitivity analysis"""
        modifications = [
            self.generate_counterfactual_glasses(image, current_similarity),
            self.generate_counterfactual_expression(image, current_similarity),
            self.generate_counterfactual_hair(image, current_similarity)
        ]
        
        # Add additional modifications
        additional_mods = [
            {
                'modification': 'Lighting Change',
                'change': -0.018,
                'confidence': 0.83,
                'explanation': 'Different lighting conditions affect facial feature visibility'
            },
            {
                'modification': 'Head Pose Change',
                'change': -0.032,
                'confidence': 0.91,
                'explanation': 'Pose variations significantly impact face recognition accuracy'
            },
            {
                'modification': 'Age Simulation (+5 years)',
                'change': -0.028,
                'confidence': 0.76,
                'explanation': 'Aging effects gradually reduce recognition performance'
            },
            {
                'modification': 'Makeup Application',
                'change': -0.015,
                'confidence': 0.82,
                'explanation': 'Makeup can alter facial features and affect recognition'
            }
        ]
        
        for mod in additional_mods:
            mod['original_similarity'] = current_similarity
            mod['new_similarity'] = max(0, min(1, current_similarity + mod['change']))
            modifications.append(mod)
        
        # Sort by impact magnitude
        modifications.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return modifications
    
    def _detect_glasses(self, image):
        """Mock glasses detection"""
        # In real implementation, this would use computer vision
        return False  # Assume no glasses for demo
    
    def _detect_expression(self, image):
        """Mock expression detection"""
        # In real implementation, this would use expression recognition
        return 'neutral'  # Assume neutral expression for demo


class PrototypeVisualizationHandler:
    """Handler for prototype visualization and analysis"""
    
    def __init__(self):
        self.prototype_database = self._create_mock_database()
    
    def _create_mock_database(self):
        """Create mock prototype database"""
        return {
            'male_young': {
                'prototypes': [
                    {
                        'id': 'male_young_001',
                        'attributes': ['Male', 'Young', 'Attractive', 'Brown_Hair', 'No_Beard'],
                        'embedding': None,  # Would contain actual embedding in production
                        'confidence': 0.94,
                        'frequency': 847,  # How often this prototype appears
                        'description': 'Common young male face pattern'
                    }
                ]
            },
            'female_young': {
                'prototypes': [
                    {
                        'id': 'female_young_001',
                        'attributes': ['Female', 'Young', 'Attractive', 'Long_Hair', 'Smiling'],
                        'embedding': None,
                        'confidence': 0.92,
                        'frequency': 723,
                        'description': 'Common young female face pattern'
                    }
                ]
            }
        }
    
    def get_prototype_explanation(self, prototype_id):
        """Get detailed explanation for a specific prototype"""
        explanations = {
            1: {
                'title': 'Young Male Professional',
                'description': 'This prototype represents a common pattern in the dataset: young professional males with clean-cut appearance.',
                'key_features': [
                    'Clean facial structure with defined jawline',
                    'Professional grooming (no facial hair)',
                    'Clear skin indicative of younger age',
                    'Standard professional lighting conditions'
                ],
                'dataset_frequency': 'Appears in ~12% of training data',
                'recognition_accuracy': '94.2% average accuracy',
                'common_variations': [
                    'With/without glasses',
                    'Different hair colors',
                    'Various expressions'
                ]
            },
            2: {
                'title': 'Expressive Young Male',
                'description': 'This prototype captures faces with positive expressions and emotional engagement.',
                'key_features': [
                    'Visible smile or positive expression',
                    'Engaged eye contact with camera',
                    'Natural, relaxed facial muscles',
                    'Good lighting that enhances features'
                ],
                'dataset_frequency': 'Appears in ~8% of training data',
                'recognition_accuracy': '91.7% average accuracy',
                'common_variations': [
                    'Intensity of smile',
                    'Eye visibility',
                    'Head angle variations'
                ]
            }
        }
        
        return explanations.get(prototype_id, {
            'title': f'Prototype #{prototype_id}',
            'description': 'Face pattern identified during model training.',
            'key_features': ['Distinctive facial structure', 'Common attribute combination'],
            'dataset_frequency': 'Variable frequency in dataset',
            'recognition_accuracy': '85-95% typical range'
        })


class CounterfactualVisualizer:
    """Handler for creating and visualizing counterfactual explanations"""
    
    def __init__(self):
        self.modification_templates = self._load_modification_templates()
    
    def _load_modification_templates(self):
        """Load templates for different types of modifications"""
        return {
            'glasses': {
                'name': 'Eyeglasses',
                'typical_impact': -0.054,
                'variance': 0.023,
                'explanation': 'Glasses create occlusion around the eye region, which is critical for face recognition.',
                'technical_details': 'The periocular region contributes ~35% to face recognition accuracy.',
                'mitigation': 'Modern systems use multiple facial regions to reduce glasses impact.'
            },
            'expression': {
                'name': 'Facial Expression',
                'typical_impact': 0.037,
                'variance': 0.015,
                'explanation': 'Expressions change facial muscle configuration and feature positions.',
                'technical_details': 'Smile detection accuracy: 89%, impacts mouth and eye regions.',
                'mitigation': 'Expression-invariant features focus on bone structure.'
            },
            'lighting': {
                'name': 'Illumination',
                'typical_impact': -0.029,
                'variance': 0.041,
                'explanation': 'Lighting affects shadow patterns and feature visibility.',
                'technical_details': 'Face recognition drops 15-25% under poor lighting conditions.',
                'mitigation': 'Normalization techniques reduce lighting sensitivity.'
            }
        }
    
    def create_comparison_view(self, original_score, counterfactual_score, modification_type):
        """Create a visual comparison of before/after scenarios"""
        template = self.modification_templates.get(modification_type, {})
        
        return {
            'original': {
                'score': original_score,
                'label': 'Original Image',
                'confidence': 'High' if original_score > 0.7 else 'Medium' if original_score > 0.5 else 'Low'
            },
            'modified': {
                'score': counterfactual_score,
                'label': f'With {template.get("name", modification_type)}',
                'confidence': 'High' if counterfactual_score > 0.7 else 'Medium' if counterfactual_score > 0.5 else 'Low'
            },
            'analysis': {
                'change': counterfactual_score - original_score,
                'percentage_change': ((counterfactual_score - original_score) / original_score) * 100,
                'direction': 'Improved' if counterfactual_score > original_score else 'Degraded',
                'magnitude': 'Large' if abs(counterfactual_score - original_score) > 0.05 else 'Medium' if abs(counterfactual_score - original_score) > 0.02 else 'Small'
            },
            'explanation': template.get('explanation', ''),
            'technical_details': template.get('technical_details', ''),
            'mitigation': template.get('mitigation', '')
        }
    
    def get_feature_importance_map(self, modification_type):
        """Get feature importance map for specific modification"""
        importance_maps = {
            'glasses': {
                'eye_region': 0.89,
                'nose_bridge': 0.67,
                'upper_face': 0.45,
                'lower_face': 0.23,
                'hair_region': 0.12
            },
            'expression': {
                'mouth_region': 0.92,
                'eye_region': 0.76,
                'cheek_region': 0.54,
                'forehead': 0.31,
                'chin_region': 0.48
            },
            'lighting': {
                'overall_face': 0.88,
                'shadow_regions': 0.71,
                'highlight_regions': 0.63,
                'texture_details': 0.45,
                'edge_definition': 0.39
            }
        }
        
        return importance_maps.get(modification_type, {})


# JavaScript integration helpers
def generate_interactive_js_config():
    """Generate JavaScript configuration for interactive features"""
    return {
        'prototype_config': {
            'max_prototypes': 5,
            'similarity_threshold': 0.7,
            'display_attributes': 5,
            'animation_duration': 300
        },
        'counterfactual_config': {
            'available_modifications': ['glasses', 'expression', 'hair', 'lighting'],
            'sensitivity_threshold': 0.02,
            'confidence_threshold': 0.8,
            'visualization_mode': 'side_by_side'
        },
        'accessibility_config': {
            'screen_reader_descriptions': True,
            'keyboard_navigation': True,
            'high_contrast_mode': True,
            'text_alternative_explanations': True
        }
    }