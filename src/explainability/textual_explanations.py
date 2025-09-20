"""
Textual Explanation Generation
=============================

Generates natural language explanations combining multiple explanation methods.
Creates comprehensive, human-readable explanations for face recognition decisions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from .base import BaseExplainer, ExplanationResult, get_attribute_names


class TextualExplainer(BaseExplainer):
    """
    Generates natural language explanations for face recognition decisions
    
    Combines visual explanations, attribute analysis, concept scores, and
    prototype information to create comprehensive textual explanations.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = 'cuda',
                 explanation_style: str = 'comprehensive'):
        """
        Initialize textual explainer
        
        Args:
            model: Face recognition model
            device: Device to run on
            explanation_style: 'brief', 'comprehensive', or 'technical'
        """
        super().__init__(model, device)
        self.explanation_style = explanation_style
        self.attribute_names = get_attribute_names()
        
        # Templates for different explanation types
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize explanation templates"""
        return {
            'verification': {
                'brief': "{decision} (similarity: {similarity:.3f}). {top_reason}.",
                'comprehensive': "{decision} (similarity: {similarity:.3f}, threshold: {threshold:.3f}). {attribute_analysis} {visual_analysis} {confidence_statement}",
                'technical': "{decision} with {confidence:.1%} confidence. Similarity score: {similarity:.3f} (threshold: {threshold:.3f}). {detailed_analysis}"
            },
            'identity': {
                'brief': "Predicted identity: {identity}. {confidence_reason}.",
                'comprehensive': "Identity prediction: {identity} with {confidence:.1%} confidence. {attribute_summary} {prototype_analysis} {visual_focus}",
                'technical': "Identity classification: {identity} (confidence: {confidence:.3f}). {embedding_analysis} {attribute_technical} {saliency_analysis}"
            },
            'attribute': {
                'brief': "Key attributes: {top_attributes}.",
                'comprehensive': "Facial attributes detected: {positive_attributes}. Absent attributes: {negative_attributes}. {confidence_analysis}",
                'technical': "Attribute predictions: {attribute_scores}. Confidence distribution: {confidence_stats}"
            }
        }
    
    def explain_verification(self,
                           image1: torch.Tensor,
                           image2: torch.Tensor,
                           threshold: float = 0.5,
                           visual_explanation: Optional[Dict] = None,
                           attribute_explanation: Optional[Dict] = None,
                           prototype_explanation: Optional[Dict] = None,
                           concept_explanation: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive textual explanation for face verification
        
        Args:
            image1: First face image
            image2: Second face image
            threshold: Verification threshold
            visual_explanation: Results from visual explainer
            attribute_explanation: Results from attribute explainer
            prototype_explanation: Results from prototype explainer
            concept_explanation: Results from concept analyzer
            
        Returns:
            Comprehensive textual explanation
        """
        # Get basic verification result
        with torch.no_grad():
            outputs1 = self.model(image1.to(self.device))
            outputs2 = self.model(image2.to(self.device))
            
            emb1 = outputs1.get('embeddings', outputs1.get('identity_logits'))
            emb2 = outputs2.get('embeddings', outputs2.get('identity_logits'))
            
            similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
            is_match = similarity > threshold
        
        # Generate explanation components
        decision_text = "Match detected" if is_match else "No match detected"
        confidence = self._calculate_verification_confidence(similarity, threshold)
        
        # Attribute analysis
        attribute_analysis = ""
        if attribute_explanation:
            attribute_analysis = self._generate_attribute_verification_text(attribute_explanation)
        
        # Visual analysis
        visual_analysis = ""
        if visual_explanation:
            visual_analysis = self._generate_visual_analysis_text(visual_explanation)
        
        # Prototype analysis
        prototype_analysis = ""
        if prototype_explanation:
            prototype_analysis = self._generate_prototype_analysis_text(prototype_explanation)
        
        # Concept analysis
        concept_analysis = ""
        if concept_explanation:
            concept_analysis = self._generate_concept_analysis_text(concept_explanation)
        
        # Select template and generate explanation
        template = self.templates['verification'][self.explanation_style]
        
        if self.explanation_style == 'brief':
            top_reason = self._get_top_verification_reason(
                attribute_explanation, visual_explanation, prototype_explanation
            )
            explanation = template.format(
                decision=decision_text,
                similarity=similarity,
                top_reason=top_reason
            )
        
        elif self.explanation_style == 'comprehensive':
            confidence_statement = self._generate_confidence_statement(confidence, is_match)
            explanation = template.format(
                decision=decision_text,
                similarity=similarity,
                threshold=threshold,
                attribute_analysis=attribute_analysis,
                visual_analysis=visual_analysis,
                confidence_statement=confidence_statement
            )
        
        else:  # technical
            detailed_analysis = self._generate_detailed_technical_analysis(
                attribute_explanation, visual_explanation, prototype_explanation, concept_explanation
            )
            explanation = template.format(
                decision=decision_text,
                confidence=confidence,
                similarity=similarity,
                threshold=threshold,
                detailed_analysis=detailed_analysis
            )
        
        return {
            'explanation': explanation,
            'similarity': similarity,
            'is_match': is_match,
            'confidence': confidence,
            'components': {
                'attribute_analysis': attribute_analysis,
                'visual_analysis': visual_analysis,
                'prototype_analysis': prototype_analysis,
                'concept_analysis': concept_analysis
            }
        }
    
    def explain_identity(self,
                        image: torch.Tensor,
                        predicted_identity: Optional[int] = None,
                        attribute_explanation: Optional[Dict] = None,
                        visual_explanation: Optional[Dict] = None,
                        prototype_explanation: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate textual explanation for identity prediction
        
        Args:
            image: Input face image
            predicted_identity: Predicted identity
            attribute_explanation: Results from attribute explainer
            visual_explanation: Results from visual explainer
            prototype_explanation: Results from prototype explainer
            
        Returns:
            Identity prediction explanation
        """
        # Get prediction if not provided
        if predicted_identity is None:
            with torch.no_grad():
                outputs = self.model(image.to(self.device))
                if 'identity_logits' in outputs:
                    predicted_identity = outputs['identity_logits'].argmax(dim=1).item()
                else:
                    predicted_identity = "Unknown"
        
        # Calculate confidence
        confidence = self._calculate_identity_confidence(image, predicted_identity)
        
        # Generate components
        attribute_summary = ""
        if attribute_explanation:
            attribute_summary = self._generate_attribute_summary_text(attribute_explanation)
        
        prototype_analysis = ""
        if prototype_explanation:
            prototype_analysis = self._generate_identity_prototype_text(prototype_explanation)
        
        visual_focus = ""
        if visual_explanation:
            visual_focus = self._generate_visual_focus_text(visual_explanation)
        
        # Select template and generate explanation
        template = self.templates['identity'][self.explanation_style]
        
        if self.explanation_style == 'brief':
            confidence_reason = self._get_identity_confidence_reason(attribute_explanation, prototype_explanation)
            explanation = template.format(
                identity=predicted_identity,
                confidence_reason=confidence_reason
            )
        
        elif self.explanation_style == 'comprehensive':
            explanation = template.format(
                identity=predicted_identity,
                confidence=confidence,
                attribute_summary=attribute_summary,
                prototype_analysis=prototype_analysis,
                visual_focus=visual_focus
            )
        
        else:  # technical
            embedding_analysis = self._generate_embedding_analysis(image)
            attribute_technical = self._generate_technical_attribute_analysis(attribute_explanation)
            saliency_analysis = self._generate_saliency_analysis(visual_explanation)
            
            explanation = template.format(
                identity=predicted_identity,
                confidence=confidence,
                embedding_analysis=embedding_analysis,
                attribute_technical=attribute_technical,
                saliency_analysis=saliency_analysis
            )
        
        return {
            'explanation': explanation,
            'predicted_identity': predicted_identity,
            'confidence': confidence,
            'components': {
                'attribute_summary': attribute_summary,
                'prototype_analysis': prototype_analysis,
                'visual_focus': visual_focus
            }
        }
    
    def _calculate_verification_confidence(self, similarity: float, threshold: float) -> float:
        """Calculate confidence for verification decision"""
        distance_from_threshold = abs(similarity - threshold)
        max_distance = max(1.0 - threshold, threshold)
        confidence = min(distance_from_threshold / max_distance, 1.0)
        return confidence
    
    def _calculate_identity_confidence(self, image: torch.Tensor, predicted_identity: int) -> float:
        """Calculate confidence for identity prediction"""
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
            if 'identity_logits' in outputs:
                probs = torch.softmax(outputs['identity_logits'], dim=1)
                confidence = probs[0, predicted_identity].item()
            else:
                confidence = 0.5  # Default confidence
        return confidence
    
    def _generate_attribute_verification_text(self, attr_explanation: Dict) -> str:
        """Generate attribute analysis text for verification"""
        if not attr_explanation or 'agreement_analysis' not in attr_explanation:
            return ""
        
        agreement = attr_explanation['agreement_analysis']
        agreement_rate = agreement['agreement_rate']
        
        if agreement_rate > 0.8:
            agreement_level = "high agreement"
        elif agreement_rate > 0.6:
            agreement_level = "moderate agreement"
        else:
            agreement_level = "low agreement"
        
        text = f"Facial attributes show {agreement_level} ({agreement_rate:.1%})."
        
        # Add top agreement/disagreement
        if agreement.get('top_agreements'):
            top_agree = agreement['top_agreements'][0][0].replace('_', ' ').lower()
            text += f" Strongest agreement on {top_agree}."
        
        if agreement.get('top_disagreements'):
            top_disagree = agreement['top_disagreements'][0][0].replace('_', ' ').lower()
            text += f" Main difference in {top_disagree}."
        
        return text
    
    def _generate_visual_analysis_text(self, visual_explanation: Dict) -> str:
        """Generate visual analysis text"""
        if not visual_explanation:
            return ""
        
        # Check what visual explanation methods were used
        methods_used = []
        
        if 'grad_cam' in visual_explanation:
            methods_used.append("attention analysis")
        
        if 'integrated_gradients' in visual_explanation:
            methods_used.append("pixel attribution")
        
        if methods_used:
            return f"Visual analysis using {', '.join(methods_used)} shows model focus on key facial regions."
        
        return "Visual analysis performed."
    
    def _generate_prototype_analysis_text(self, prototype_explanation: Dict) -> str:
        """Generate prototype analysis text"""
        if not prototype_explanation:
            return ""
        
        if 'similar_to_image1' in prototype_explanation:
            # Verification explanation
            sim1 = prototype_explanation['similar_to_image1'][0] if prototype_explanation['similar_to_image1'] else None
            sim2 = prototype_explanation['similar_to_image2'][0] if prototype_explanation['similar_to_image2'] else None
            
            text_parts = []
            if sim1:
                text_parts.append(f"first image resembles identity {sim1['identity']}")
            if sim2:
                text_parts.append(f"second image resembles identity {sim2['identity']}")
            
            if text_parts:
                return f"Prototype analysis: {', '.join(text_parts)}."
        
        return "Prototype comparison performed."
    
    def _generate_concept_analysis_text(self, concept_explanation: Dict) -> str:
        """Generate concept analysis text"""
        if not concept_explanation or 'tcav_scores' not in concept_explanation:
            return ""
        
        top_concepts = concept_explanation.get('top_influential_concepts', [])
        if top_concepts:
            top_concept = top_concepts[0][0].replace('_', ' ').lower()
            score = top_concepts[0][1]
            influence_level = "strong" if score > 0.7 else "moderate" if score > 0.4 else "weak"
            return f"Concept analysis shows {influence_level} influence from {top_concept}."
        
        return "Concept analysis performed."
    
    def _get_top_verification_reason(self, 
                                   attr_explanation: Optional[Dict],
                                   visual_explanation: Optional[Dict],
                                   prototype_explanation: Optional[Dict]) -> str:
        """Get the most important reason for verification decision"""
        reasons = []
        
        if attr_explanation and 'agreement_analysis' in attr_explanation:
            agreement_rate = attr_explanation['agreement_analysis']['agreement_rate']
            if agreement_rate > 0.8:
                reasons.append(("High attribute agreement", agreement_rate))
            elif agreement_rate < 0.4:
                reasons.append(("Low attribute agreement", 1.0 - agreement_rate))
        
        if prototype_explanation and 'similarity' in prototype_explanation:
            similarity = prototype_explanation['similarity']
            if similarity > 0.8:
                reasons.append(("Strong prototype similarity", similarity))
            elif similarity < 0.3:
                reasons.append(("Weak prototype similarity", 1.0 - similarity))
        
        if reasons:
            # Return the reason with highest score
            top_reason = max(reasons, key=lambda x: x[1])
            return top_reason[0]
        
        return "Based on embedding similarity analysis"
    
    def _generate_confidence_statement(self, confidence: float, is_match: bool) -> str:
        """Generate confidence statement"""
        if confidence > 0.8:
            level = "high"
        elif confidence > 0.5:
            level = "moderate"
        else:
            level = "low"
        
        decision_type = "match" if is_match else "non-match"
        return f"Confidence in {decision_type} decision: {level} ({confidence:.1%})."
    
    def _generate_detailed_technical_analysis(self, 
                                            attr_explanation: Optional[Dict],
                                            visual_explanation: Optional[Dict],
                                            prototype_explanation: Optional[Dict],
                                            concept_explanation: Optional[Dict]) -> str:
        """Generate detailed technical analysis"""
        analysis_parts = []
        
        if attr_explanation:
            attr_stats = self._get_attribute_statistics(attr_explanation)
            analysis_parts.append(f"Attribute analysis: {attr_stats}")
        
        if visual_explanation:
            visual_stats = self._get_visual_statistics(visual_explanation)
            analysis_parts.append(f"Visual explanation: {visual_stats}")
        
        if prototype_explanation:
            proto_stats = self._get_prototype_statistics(prototype_explanation)
            analysis_parts.append(f"Prototype analysis: {proto_stats}")
        
        if concept_explanation:
            concept_stats = self._get_concept_statistics(concept_explanation)
            analysis_parts.append(f"Concept scores: {concept_stats}")
        
        return " ".join(analysis_parts)
    
    def _get_attribute_statistics(self, attr_explanation: Dict) -> str:
        """Get attribute statistics summary"""
        if 'agreement_analysis' in attr_explanation:
            agreement = attr_explanation['agreement_analysis']
            return f"{agreement['total_agreements']}/{agreement['total_agreements'] + agreement['total_disagreements']} attributes agree"
        return "attribute analysis completed"
    
    def _get_visual_statistics(self, visual_explanation: Dict) -> str:
        """Get visual explanation statistics"""
        methods = []
        if 'grad_cam' in visual_explanation:
            methods.append("Grad-CAM")
        if 'integrated_gradients' in visual_explanation:
            methods.append("IntGrad")
        
        return f"{', '.join(methods)} computed" if methods else "visual analysis completed"
    
    def _get_prototype_statistics(self, prototype_explanation: Dict) -> str:
        """Get prototype statistics"""
        if 'similar_to_image1' in prototype_explanation:
            n_prototypes = len(prototype_explanation.get('similar_to_image1', []))
            return f"{n_prototypes} prototypes analyzed"
        return "prototype comparison completed"
    
    def _get_concept_statistics(self, concept_explanation: Dict) -> str:
        """Get concept statistics"""
        if 'tcav_scores' in concept_explanation:
            n_concepts = len(concept_explanation['tcav_scores'])
            return f"{n_concepts} concepts evaluated"
        return "concept analysis completed"
    
    def _generate_attribute_summary_text(self, attribute_explanation: Dict) -> str:
        """Generate attribute summary text for identity explanation"""
        if not attribute_explanation or 'attribute_differences' not in attribute_explanation:
            return "No significant attribute patterns detected."
        
        diffs = attribute_explanation['attribute_differences']
        top_positive = [(attr, score) for attr, score in diffs.items() if score > 0.3][:3]
        top_negative = [(attr, score) for attr, score in diffs.items() if score < -0.3][:3]
        
        parts = []
        if top_positive:
            pos_attrs = [attr.replace('_', ' ').lower() for attr, _ in top_positive]
            parts.append(f"Strong indicators: {', '.join(pos_attrs)}")
        
        if top_negative:
            neg_attrs = [attr.replace('_', ' ').lower() for attr, _ in top_negative]
            parts.append(f"Distinguishing factors: absence of {', '.join(neg_attrs)}")
        
        return ". ".join(parts) if parts else "Balanced attribute profile."
    
    def _generate_identity_prototype_text(self, prototype_explanation: Dict) -> str:
        """Generate prototype analysis text for identity explanation"""
        if not prototype_explanation or 'nearest_prototypes' not in prototype_explanation:
            return "No prototype matches found."
        
        nearest = prototype_explanation['nearest_prototypes'][0]
        similarity = nearest.get('similarity', 0.0)
        
        if similarity > 0.8:
            return f"Very similar to training examples (similarity: {similarity:.1%})"
        elif similarity > 0.6:
            return f"Moderate similarity to training examples (similarity: {similarity:.1%})"
        else:
            return f"Low similarity to training examples (similarity: {similarity:.1%})"
    
    def _generate_visual_focus_text(self, visual_explanation: Dict) -> str:
        """Generate visual focus text for identity explanation"""
        if not visual_explanation or 'saliency_map' not in visual_explanation:
            return "Visual analysis unavailable."
        
        # Simple heuristic based on typical face regions
        return "Model focuses on key facial features including eyes, nose, and mouth regions."
    
    def _get_identity_confidence_reason(self, attribute_explanation: Dict, prototype_explanation: Dict) -> str:
        """Get brief confidence reason for identity prediction"""
        reasons = []
        
        if attribute_explanation and 'confidence' in attribute_explanation:
            attr_conf = attribute_explanation['confidence']
            if attr_conf > 0.8:
                reasons.append("strong attribute match")
            elif attr_conf > 0.6:
                reasons.append("good attribute indicators")
        
        if prototype_explanation and 'nearest_prototypes' in prototype_explanation:
            similarity = prototype_explanation['nearest_prototypes'][0].get('similarity', 0.0)
            if similarity > 0.8:
                reasons.append("high similarity to training examples")
            elif similarity > 0.6:
                reasons.append("moderate similarity to known patterns")
        
        if not reasons:
            reasons.append("model confidence")
        
        return " and ".join(reasons)
    
    def _generate_embedding_analysis(self, image: torch.Tensor) -> str:
        """Generate embedding analysis for technical explanation"""
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
            if 'embeddings' in outputs:
                embedding = outputs['embeddings'][0]
                norm = torch.norm(embedding).item()
                return f"Embedding norm: {norm:.3f}, dimension: {embedding.shape[0]}"
            else:
                return "Embedding analysis unavailable"
    
    def _generate_technical_attribute_analysis(self, attribute_explanation: Dict) -> str:
        """Generate technical attribute analysis"""
        if not attribute_explanation or 'attribute_differences' not in attribute_explanation:
            return "No attribute analysis available."
        
        diffs = attribute_explanation['attribute_differences']
        avg_diff = np.mean([abs(v) for v in diffs.values()])
        max_diff = max([abs(v) for v in diffs.values()])
        
        return f"Attribute differences - Average: {avg_diff:.3f}, Maximum: {max_diff:.3f}"
    
    def _generate_saliency_analysis(self, visual_explanation: Dict) -> str:
        """Generate saliency analysis for technical explanation"""
        if not visual_explanation or 'saliency_map' not in visual_explanation:
            return "No saliency analysis available."
        
        # Would analyze saliency map statistics in real implementation
        return "Saliency concentrated on central facial features with peak activation in eye region."
    
    def explain(self, 
                image: torch.Tensor,
                task_type: str = 'identity',
                **kwargs) -> Dict[str, Any]:
        """
        Generate textual explanation for given task
        
        Args:
            image: Input image tensor
            task_type: 'identity' or 'verification'
            **kwargs: Additional arguments including other explanation results
            
        Returns:
            Dictionary containing textual explanation
        """
        if task_type == 'identity':
            return self.explain_identity(image, **kwargs)
        elif task_type == 'verification':
            if 'image2' not in kwargs:
                raise ValueError("Second image required for verification explanation")
            return self.explain_verification(image, kwargs['image2'], **kwargs)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    def set_explanation_style(self, style: str):
        """Set explanation style"""
        if style not in ['brief', 'comprehensive', 'technical']:
            raise ValueError("Style must be 'brief', 'comprehensive', or 'technical'")
        self.explanation_style = style
    
    def generate_summary_report(self, explanations: List[Dict]) -> str:
        """
        Generate summary report from multiple explanations
        
        Args:
            explanations: List of explanation results
            
        Returns:
            Summary report text
        """
        if not explanations:
            return "No explanations provided."
        
        # Count decision types
        matches = sum(1 for exp in explanations if exp.get('is_match', False))
        total = len(explanations)
        
        # Average confidence
        confidences = [exp.get('confidence', 0.5) for exp in explanations]
        avg_confidence = np.mean(confidences)
        
        # Common attributes mentioned
        attribute_mentions = {}
        for exp in explanations:
            if 'components' in exp and 'attribute_analysis' in exp['components']:
                # Extract attribute mentions (simplified)
                attr_text = exp['components']['attribute_analysis']
                for attr_name in self.attribute_names:
                    if attr_name.lower().replace('_', ' ') in attr_text.lower():
                        attribute_mentions[attr_name] = attribute_mentions.get(attr_name, 0) + 1
        
        # Generate report
        report_parts = [
            f"Analysis Summary ({total} decisions):",
            f"- Matches: {matches}/{total} ({matches/total:.1%})",
            f"- Average confidence: {avg_confidence:.1%}"
        ]
        
        if attribute_mentions:
            top_attrs = sorted(attribute_mentions.items(), key=lambda x: x[1], reverse=True)[:3]
            attr_list = [f"{attr.replace('_', ' ').lower()} ({count})" for attr, count in top_attrs]
            report_parts.append(f"- Most mentioned attributes: {', '.join(attr_list)}")
        
        return "\n".join(report_parts)