# Nombre del entorno virtual
VENV = venv

# Crear el entorno virtual
venv:
	python -m venv $(VENV)

# Generar el archivo requirements.txt
requirements: venv
	$(VENV)\Scripts\python -m pip freeze > requirements.txt

# Instalar dependencias desde requirements.txt
install: venv
	$(VENV)\Scripts\python -m pip install -r requirements.txt

# Limpiar el entorno virtual y requirements.txt
clean:
	@if exist $(VENV) (rmdir /s /q $(VENV))
	@if exist requirements.txt (del requirements.txt)
