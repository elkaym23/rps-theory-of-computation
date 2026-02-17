from flask import Flask, render_template, request, jsonify, Response, url_for
import random
import atexit
import os
from openai import OpenAI
from dotenv import load_dotenv
from camera import Camera

app = Flask(__name__)

# --------------------------
# AI Suggestions Function
# --------------------------
# Initialize OpenAI client

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global list to store CPU history
cpu_history = []
def predict_rps_next_move(history):
    
    # Convert history list to a string for the prompt
    history_str = ', '.join(history)
    
    prompt = f"""
    You are an AI that predicts the next move in Rock-Paper-Scissors.
    Given this sequence: [{history_str}], predict the next move (rock, paper, or scissors) and what the user should play to win in one sentence.
    The format of the sentence should be AI predicts the next move will be --- so you should play --- to win
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # deterministic
    )
    
    # Extract the predicted move from the response
    predicted_move = response.choices[0].message.content.strip().upper()
    
    return predicted_move

# Create camera instance ONCE
cam = Camera()


# --------------------------
# Shutdown Handler (Prevents Camera Lock)
# --------------------------
@atexit.register
def shutdown_camera():
    if cam.cap.isOpened():
        cam.cap.release()


# --------------------------
# Routes
# --------------------------

@app.route("/")
def start():
    return render_template("start.html")


@app.route("/countdown")
def countdown():
    return render_template("countdown.html")


@app.route("/gameforcam")
def gameforcam():
    return render_template("gameforcam.html")

@app.route("/cam_countdown")
def cam_countdown():
    return render_template("cam_countdown.html")

# --multiplayer---
@app.route("/multiplayer")
def multiplayer():
    return render_template("multiplayer.html")


@app.route("/pvp_simul")
def pvp_simul():
    return render_template("pvp_simul.html")

@app.route("/pvp_result")
def pvp_result():
    p1 = request.args.get("p1")
    p2 = request.args.get("p2")

    if p1 == p2:
        result = "IT'S A DRAW!"
    elif (p1 == "rock" and p2 == "scissors") or \
         (p1 == "paper" and p2 == "rock") or \
         (p1 == "scissors" and p2 == "paper"):
        result = "PLAYER 1 WINS!"
    else:
        result = "PLAYER 2 WINS!"

    return render_template("pvp_result.html", p1=p1, p2=p2, result=result)


# --------------------------
# 1. Video Feed Route
# --------------------------
@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --------------------------
# 2. Prediction Route
# --------------------------
@app.route('/predict')
def predict():
    result = cam.predict()
    return jsonify(result)


# --------------------------
# 3. Classic RPS Results Page
# --------------------------
@app.route("/result")
def result():
    global cpu_history
    player = request.args.get("player")

    if player not in ["rock", "paper", "scissors"]:
        # fallback: avoid crashes if user manually tampers URL
        player = random.choice(["rock", "paper", "scissors"])

    cpu = random.choice(["rock", "paper", "scissors"])

    # Save CPU move to history
    cpu_history.append(cpu)
    cpu_history[:] = cpu_history[-10:]  # saves only the last 10 moves

    # Calling prediction function using CPU history
    predicted_cpu_move = predict_rps_next_move(cpu_history)

    # determine winner
    if player == cpu:
        result_text = "IT'S A DRAW!"
    elif (player == "rock" and cpu == "scissors") or \
         (player == "paper" and cpu == "rock") or \
         (player == "scissors" and cpu == "paper"):
        result_text = "YOU WIN!"
    else:
        result_text = "YOU LOSE!"

    return render_template(
        "result.html",
        player=player,
        cpu=cpu,
        predicted_cpu_move=predicted_cpu_move,
        result_text=result_text
    )


@app.route("/cam_result")
def cam_result():
    user = request.args.get("user")
    ai = request.args.get("ai")

    rock = round(float(request.args.get("rock")), 3)
    paper = round(float(request.args.get("paper")), 3)
    scissors = round(float(request.args.get("scissors")), 3)
    invalid = round(float(request.args.get("invalid", 0)), 3)

    # --- HANDLE INVALID FIRST ---
    if user == "invalid":
        result_text = "INVALID MOVE"
        ai = "none"   # CPU should not show a hand
    else:
        # --- NORMAL GAME LOGIC ---
        if user == ai:
            result_text = "IT'S A DRAW!"
        elif (user == "rock" and ai == "scissors") or \
             (user == "paper" and ai == "rock") or \
             (user == "scissors" and ai == "paper"):
            result_text = "YOU WIN!"
        else:
            result_text = "YOU LOSE!"

    return render_template(
        "cam_result.html",
        user=user,
        ai=ai,
        rock=rock,
        paper=paper,
        scissors=scissors,
        invalid=invalid,
        result_text=result_text
    )

# --------------------------
# Start Flask Server
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
