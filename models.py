import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()




class Puissance(db.Model):
    __tablename__ = "carpuissance"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    #flight_id = db.Column(db.Integer, db.ForeignKey("flights.id"), nullable=False)
 
class CarModel(db.Model):
    __tablename__ = "carmodel"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    
class CarUsage(db.Model):
    __tablename__ = "carusage"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)

class Car(db.Model):
    __tablename__ = "car"
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String, nullable=False)
    energie = db.Column(db.String, nullable=False)
    puissance = db.Column(db.String, nullable=False)
    usage = db.Column(db.String, nullable=False)
    
    typepolice = db.Column(db.String, nullable=False)
    naturepolice = db.Column(db.String, nullable=False)
    bonusmalus = db.Column(db.String, nullable=False)
    sinistrecount = db.Column(db.Integer, nullable=False)
    SumResponsabilite = db.Column(db.Integer, nullable=False)
    licence = db.Column(db.Integer, nullable=False)
    
    target = db.Column(db.Integer, nullable=False)
    
 


