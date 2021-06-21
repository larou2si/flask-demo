import pandas as pd
import os
from flask import Flask, jsonify, render_template, request, session
from flask_session import Session

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
#app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:gogeo@localhost/pfe"
#app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
#db.init_app(app)

#socketio = SocketIO(app)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
#
## os.environ['DATABASE_URL'] = "postgres://postgres:root@localhost/projetML"
## os.getenv("DATABASE_URL")
##engine = create_engine("postgresql://postgres:gogeo@localhost/pfe")
##db = scoped_session(sessionmaker(bind=engine))
#
#
@app.route("/")
def index():

    if request.method == "POST":
        filename = request.form.get("filename")
        return jsonify({"success": True, "df": session[filename].to_})
    data = {
            }
    return render_template("mlground_index.html", data=data)
#
#
## ============================ projet datascince: insurance===========================================
#
#@app.route("/police0")
#def police0():
#    return render_template("view_police.html")
#
#
#@app.route("/police1", methods=["GET"])
#def police1():
#    #  return jsonify({"success": False})
#    return jsonify({"success": True, "rate": "POLICE"})
#
#    # Predict: saisir le form pour calculer
#
#
#@app.route("/mlform", methods=["GET", "POST"])
#def mlform():
#    carmodels = CarModel.query.all()
#    puisances = Puissance.query.all()
#    usages = CarUsage.query.all()
#    return render_template("ml_form.html", carmodels=carmodels, puisances=puisances,
#                           usages=usages)
#
#
#@app.route("/carform", methods=["POST"])
#def carform():
#    model = request.form.get("model_id")
#    energie = request.form.get("energie_id")
#    puissance = request.form.get("puisance_id")
#    usage = request.form.get("usage_id")
#
#    type_police = request.form.get("type_police")
#    nature_police = request.form.get("nature_police")
#    bonusmalus = request.form.get("bonusmalus")
#    sinistrecount = request.form.get("sinistrecount")
#    SumResponsabilite = request.form.get("sumResponsabilite")
#    licence = request.form.get("licence")
#    target = request.form.get("target")
#
#    car = Car(model=model, energie=energie, puissance=puissance, usage=usage,
#              typepolice=type_police, naturepolice=nature_police, bonusmalus=bonusmalus,
#              sinistrecount=sinistrecount, SumResponsabilite=SumResponsabilite, licence=licence)
#    data = []
#    return render_template("ml_form.html", carform=car)
#
#
## ============================ end projet datascince: insurance===========================================
#@app.route("/book", methods=["POST"])
#def book():
#    """Book a flight."""
#
#    # Get form information.
#    name = request.form.get("name")
#    try:
#        flight_id = int(request.form.get("flight_id"))
#    except ValueError:
#        return render_template("error.html", message="Invalid flight number.")
#
#    # Make sure flight exists.
#    if db.execute("SELECT * FROM flights WHERE id = :id", {"id": flight_id}).rowcount == 0:
#        return render_template("error.html", message="No such flight with that id.")
#    db.execute("INSERT INTO passengers (name, flight_id) VALUES (:name, :flight_id)",
#               {"name": name, "flight_id": flight_id})
#    db.commit()
#    return render_template("success.html")
#
#
#@app.route("/flights")
#def flights():
#    """Lists all flights."""
#    flights = db.execute("SELECT * FROM flights").fetchall()
#    return render_template("flights.html", flights=flights)
#
#
#@app.route("/flights/<int:flight_id>")
#def flight(flight_id):
#    """Lists details about a single flight."""
#
#    # Make sure flight exists.
#    flight = db.execute("SELECT * FROM flights WHERE id = :id", {"id": flight_id}).fetchone()
#    if flight is None:
#        return render_template("error.html", message="No such flight.")
#
#    # Get all passengers.
#    passengers = db.execute("SELECT name FROM passengers WHERE flight_id = :flight_id",
#                            {"flight_id": flight_id}).fetchall()
#    return render_template("flight.html", flight=flight, passengers=passengers)
#
#
#votes = {"yes": 0, "no": 0, "maybe": 0}
#
#
#@app.route("/testsockets")
#def testsockets():
#    return render_template("testsockets.html", votes=votes)
#
#
#@socketio.on("submit vote")
#def vote(data):
#    selection = data["selection"]
#    votes[selection] += 1
#    emit("vote totals", votes, broadcast=True)
#

if __name__ == '__main__':
    app.run(debug=True)
    # socketio.run(app)
