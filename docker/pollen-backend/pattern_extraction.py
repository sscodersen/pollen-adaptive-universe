
from datetime import datetime
from typing import Dict, Any, List

def update_patterns(user_memory: Dict[str, Any], input_words: List[str], input_text: str, output_text: str, mode: str):
    for word in input_words:
        pattern_key = f"{mode}:{word}"
        if pattern_key not in user_memory['long_term']:
            user_memory['long_term'][pattern_key] = {
                'weight': 1.0,
                'count': 1,
                'last_seen': datetime.utcnow().isoformat(),
                'mode': mode,
                'examples': []
            }
        else:
            user_memory['long_term'][pattern_key]['weight'] += 0.1
            user_memory['long_term'][pattern_key]['count'] += 1
            user_memory['long_term'][pattern_key]['last_seen'] = datetime.utcnow().isoformat()
        # Store example interactions (max 3)
        examples = user_memory['long_term'][pattern_key]['examples']
        if len(examples) < 3:
            examples.append({
                'input': input_text[:100],
                'output': output_text[:100],
                'timestamp': datetime.utcnow().isoformat()
            })

def decay_patterns(user_memory: Dict[str, Any]):
    current_time = datetime.utcnow()
    patterns_to_remove = []
    for pattern_key, pattern_data in user_memory['long_term'].items():
        last_seen = datetime.fromisoformat(pattern_data['last_seen'])
        days_since_seen = (current_time - last_seen).days
        if days_since_seen > 7:
            pattern_data['weight'] *= 0.9
        if pattern_data['weight'] < 0.1:
            patterns_to_remove.append(pattern_key)
    for pattern_key in patterns_to_remove:
        del user_memory['long_term'][pattern_key]
