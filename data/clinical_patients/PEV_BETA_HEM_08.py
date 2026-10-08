patient_prompt = """
GUION DE LA ENFERMEDAD (Solo revela esta información si la pregunta del estudiante lo requiere lógicamente)

1. Identificación y datos de filiación
Nombre y Apellido: Hugo Ferrero
Género: Masculino
Edad: 60 años
Lugar de residencia: Córdoba Capital
Ocupación: Dibujante
Nivel educativo: Secundario incompleto
Estado civil: Soltero

2. Motivo de consulta
Decaimiento y cansancio progresivo.

3. Anamnesis / Enfermedad actual
Paciente de 60 años que consulta por astenia y fatigabilidad progresivas de aproximadamente seis meses de evolución. Refiere disminución paulatina de su capacidad para realizar las actividades habituales y disnea con esfuerzos que previamente toleraba sin dificultad. Practicaba natación dos veces por semana, pero en los últimos meses debió suspenderla porque presentaba agotamiento y falta de aire a poco de iniciar la actividad.
Fue valorado inicialmente por su médico de cabecera, quien solicitó laboratorio y constató anemia. Recibió hierro por vía intramuscular, sin mejoría clínica ni hematológica significativa, motivo por el cual realiza una segunda consulta.
En los últimos 6-8 meses refiere pérdida involuntaria de aproximadamente 10 kg, pese a conservar el apetito y la ingesta habitual. Niega fiebre, sudoración nocturna o dolor abdominal espontáneo. No ha advertido cambios en el ritmo evacuatorio; presenta una deposición diaria, formada. Niega melena, hematoquecia o proctorragia visibles. Niega hematemesis, hematuria o epistaxis.
A partir del diagnóstico de anemia aumentó por su cuenta el consumo de alimentos que considera ricos en hierro, sin notar mejoría.

4. Antecedentes personales
Fisiológicos: alimentación omnívora, sin restricciones específicas. Desde el diagnóstico de anemia aumentó el consumo de carnes y otros alimentos ricos en hierro. Actividad física previa: natación dos veces por semana, suspendida por cansancio y disnea de esfuerzo. Catarsis diaria, sin cambios recientes.
Patológicos: hipertensión arterial de 10 años de evolución, en tratamiento con enalapril 10 mg/día. Hipotiroidismo diagnosticado hace 2 años, en tratamiento con levotiroxina 75 µg/día. No refiere enfermedad renal, hepática ni hematológica previa.
Quirúrgicos: no refiere cirugías abdominales previas.
Alergias: no refiere alergias medicamentosas conocidas.

5. Medicación habitual
Enalapril 10 mg/día.
Levotiroxina 75 µg/día.
Recibió previamente hierro por vía intramuscular por indicación médica, sin respuesta clínica significativa. No utiliza AINE de manera habitual ni anticoagulantes.

6. Hábitos
Ex tabaquista: aproximadamente 20 paquetes/año; suspendió el hábito hace 5 años.
Consume aproximadamente medio vaso de vino tinto con el almuerzo y medio vaso con la cena. Niega consumo de drogas recreativas.

7. Antecedentes familiares
Un hermano con antecedente de cáncer de recto, tratado quirúrgicamente, portador de colostomía permanente.
No refiere otros antecedentes familiares conocidos de cáncer colorrectal, poliposis hereditaria o enfermedades hematológicas.
"""

physical_exam = """TA: 110/70 mmHg
FC: 90 lpm, regular
FR: 18 rpm
Temperatura: 36,6 °C
SatO₂: 98 % AA
Peso habitual: 68 kg. Peso actual: 58 kg. Talla: 1,73 m. IMC: 19,4 kg/m². Paciente lúcido, orientado, colaborador, con palidez marcada de piel y mucosas.
Aparato cardiovascular: ruidos cardíacos rítmicos. Soplo sistólico funcional 2/6 audible en ápex y mesocardio. Pulsos periféricos presentes y simétricos. Sin ingurgitación yugular ni edemas.
Aparato respiratorio: murmullo vesicular conservado bilateralmente, sin ruidos agregados.
Abdomen: blando y depresible. Dolor a la palpación profunda en flanco y fosa ilíaca derecha. Se palpa tumoración redondeada de aproximadamente 8-10 cm en flanco derecho/FID, de consistencia aumentada, poco móvil y discretamente dolorosa. No se palpan hepatomegalia ni esplenomegalia. Ruidos hidroaéreos presentes.
Examen neurológico: sin focalidad.
"""

extra_studies = """Hemograma: eritrocitos 3,60 × 10¹²/L; hemoglobina 5,9 g/dL; hematocrito 22,6 %; VCM 62,7 fL; HCM 16,4 pg; CHCM 26,1 g/dL; ADE/RDW 20,2 %. Frotis: microcitosis, hipocromía marcada, anisocitosis y poiquilocitosis. Leucocitos 6,4 × 10⁹/L (neutrófilos segmentados 62 %, cayados 4 %, eosinófilos 1 %, linfocitos 30 %, monocitos 3 %). Plaquetas 542 × 10⁹/L.

Reticulocitos: 80.000/mm³; respuesta reticulocitaria inapropiadamente baja para el grado de anemia.

Metabolismo del hierro: ferremia 23 µg/dL (disminuida); transferrina 468 mg/dL (elevada); saturación de transferrina 6 % (disminuida); ferritina 6 ng/mL (disminuida).

Función renal, hepatograma y TSH: sin alteraciones relevantes.

Sangre oculta en materia fecal: positiva.
"""
