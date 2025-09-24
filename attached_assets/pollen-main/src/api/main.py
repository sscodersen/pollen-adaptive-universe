from flask import Flask, request, jsonify
from src.models.task_proposer import TaskProposer
from src.models.task_solver import TaskSolver
from src.models.code_executor import CodeExecutor
from src.models.rl_loop import RLLoop
from src.models.memory_modules import EpisodicMemory, LongTermMemory, ContextualMemory
from src.models.ad_creation import AdCreator
from src.models.task_automation import TaskAutomation
from src.models.audio_generation import AudioGenerator, MusicGenerator
from src.models.image_generation import ImageGenerator
from src.models.video_generation import VideoGenerator
from src.models.movie_generation import MovieGenerator
from src.models.game_generation import GameGenerator
from src.models.social_post_curation import SocialPostCurator
from src.models.news_curation import NewsCurator
from src.models.trend_analysis import TrendAnalyzer

app = Flask(__name__)

@app.route('/propose-task', methods=['POST'])
def propose_task():
    data = request.get_json()
    input_text = data.get('input_text')
    proposer = TaskProposer()
    task = proposer.propose_task(input_text)
    return jsonify({"task": task})

@app.route('/solve-task', methods=['POST'])
def solve_task():
    data = request.get_json()
    input_text = data.get('input_text')
    solver = TaskSolver()
    solution = solver.solve_task(input_text)
    return jsonify({"solution": solution})

@app.route('/create-ad', methods=['POST'])
def create_ad():
    data = request.get_json()
    input_text = data.get('input_text')
    ad_creator = AdCreator()
    ad = ad_creator.create_ad(input_text)
    return jsonify({"ad": ad})

@app.route('/automate-task', methods=['POST'])
def automate_task():
    data = request.get_json()
    input_text = data.get('input_text')
    task_automation = TaskAutomation()
    result = task_automation.tasks[input_text](input_text)
    return jsonify({"result": result})

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.get_json()
    input_text = data.get('input_text')
    audio_generator = AudioGenerator()
    audio = audio_generator.generate_audio(input_text)
    return jsonify({"audio": audio})

@app.route('/generate-music', methods=['POST'])
def generate_music():
    data = request.get_json()
    input_text = data.get('input_text')
    music_generator = MusicGenerator()
    music = music_generator.integrate_with_model(input_text)
    return jsonify({"music": music})

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    input_text = data.get('input_text')
    image_generator = ImageGenerator()
    image = image_generator.generate_image(input_text)
    return jsonify({"image": image})

@app.route('/generate-video', methods=['POST'])
def generate_video():
    data = request.get_json()
    input_text = data.get('input_text')
    video_generator = VideoGenerator()
    video = video_generator.generate_video(input_text)
    return jsonify({"video": video})

@app.route('/generate-movie-script', methods=['POST'])
def generate_movie_script():
    data = request.get_json()
    input_text = data.get('input_text')
    movie_generator = MovieGenerator()
    script = movie_generator.generate_movie_script(input_text)
    return jsonify({"script": script})

@app.route('/generate-game-level', methods=['POST'])
def generate_game_level():
    data = request.get_json()
    input_text = data.get('input_text')
    game_generator = GameGenerator()
    level = game_generator.generate_game_level(input_text)
    return jsonify({"level": level})

@app.route('/curate-social-post', methods=['POST'])
def curate_social_post():
    data = request.get_json()
    input_text = data.get('input_text')
    social_post_curator = SocialPostCurator()
    post = social_post_curator.curate_post(input_text)
    return jsonify({"post": post})

@app.route('/curate-news', methods=['POST'])
def curate_news():
    data = request.get_json()
    input_text = data.get('input_text')
    news_curator = NewsCurator()
    news = news_curator.curate_news(input_text)
    return jsonify({"news": news})

@app.route('/analyze-trends', methods=['POST'])
def analyze_trends():
    data = request.get_json()
    input_text = data.get('input_text')
    trend_analyzer = TrendAnalyzer()
    trends = trend_analyzer.analyze_trends(input_text)
    return jsonify({"trends": trends})

if __name__ == '__main__':
    app.run(debug=True)