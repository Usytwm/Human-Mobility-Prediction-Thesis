# Nombre del entorno virtual
VENV = venv

# Crear el entorno virtual
venv:
	python -m venv $(VENV)

# Activar el entorno virtual y guardar las dependencias
requirements: venv
	$(VENV)\Scripts\python -m pip freeze > requirements.txt


# Instalar dependencias desde requirements.txt
install: venv
	$(VENV)/Scripts/activate && pip install -r requirements.txt

# Limpiar archivos generados
clean:
	rm -rf $(VENV) requirements.txt
