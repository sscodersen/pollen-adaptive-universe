
from typing import List, Dict, Any

def extract_adaptation_signals(feedback_data: List[Dict[str, Any]], min_confidence_threshold: float) -> Dict[str, Any]:
    signals = {
        'mode_performance': {},
        'confidence_patterns': {},
        'user_patterns': {},
        'improvement_areas': []
    }
    # Analyze performance by mode
    mode_confidences = {}
    for interaction in feedback_data:
        mode = interaction.get('mode')
        confidence = interaction.get('confidence', 0.5)
        if mode not in mode_confidences:
            mode_confidences[mode] = []
        mode_confidences[mode].append(confidence)
    # Calculate average confidence per mode
    for mode, confidences in mode_confidences.items():
        avg_confidence = sum(confidences) / len(confidences)
        signals['mode_performance'][mode] = {
            'avg_confidence': avg_confidence,
            'sample_count': len(confidences),
            'needs_improvement': avg_confidence < min_confidence_threshold
        }
        if avg_confidence < min_confidence_threshold:
            signals['improvement_areas'].append(mode)
    # Analyze user interaction patterns
    user_interactions = {}
    for interaction in feedback_data:
        user = interaction.get('user_session', 'unknown')
        if user not in user_interactions:
            user_interactions[user] = []
        user_interactions[user].append(interaction)
    signals['user_patterns'] = {
        'unique_users': len(user_interactions),
        'avg_interactions_per_user': len(feedback_data) / len(user_interactions) if user_interactions else 0,
        'most_active_users': sorted(
            user_interactions.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
    }
    return signals
