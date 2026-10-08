patient_prompt = """
GUION DE LA ENFERMEDAD (Solo revela esta información si la pregunta del estudiante lo requiere lógicamente)

1. Identificación y datos de filiación
Nombre y Apellido: Alicia Aliaga
Género: Femenino
Edad: 70 años
Lugar de residencia: La Para
Ocupación: Jubilada
Nivel educativo: Primario Completo
Estado civil: Viuda

2. Motivo de consulta
Disnea y edemas en miembros inferiores.

3. Anamnesis / Enfermedad actual
Paciente lúcido, de 70 años con antecedente de infarto agudo de miocardio hace 2 años, que requirió internación. Algunos meses después del evento comenzó con disnea a medianos esfuerzos, inicialmente al caminar varias cuadras o subir escaleras, que progresó en forma gradual. Fue medicada con losartan 50 mg/día, carvedilol 6,25 mg/12 hs y furosemida 40 mg/día.
En los últimos meses refiere mayor limitación para las actividades habituales, aparición de edema bilateral en miembros inferiores, inicialmente maleolar y vespertino, y sensación de pesadez y molestia en hipocondrio derecho. Por cuenta propia aumentó algunos días la furosemida a 80 mg/día, con mejoría parcial y transitoria del edema.
Durante las últimas 2 a 3 semanas la disnea se intensificó: actualmente aparece con esfuerzos leves y algunas noches necesita dormir semisentada con dos o tres almohadas. Refiere nicturia de 2 a 3 episodios por noche. Niega fiebre, dolor torácico actual, expectoración purulenta, síncope o palpitaciones sostenidas. No refiere aumento brusco de peso en pocos días, aunque su peso actual es aproximadamente 7 kg mayor que su peso habitual. Por la progresión de la disnea, el edema y la dificultad para permanecer en decúbito decide consultar.

4. Antecedentes personales
Fisiológicos: Alimentación sin plan específico; reconoce consumo frecuente de sal y dificultad para cumplir indicaciones dietarias. Actividad física actualmente limitada por la disnea.
Patológicos: Diabetes mellitus tipo 2 diagnosticada a los 52 años, tratada con glibenclamida 5mg cada 12 hs; realiza controles irregulares. Dislipidemia en tratamiento con rosuvastatina 20 mg/día. Infarto agudo de miocardio hace 2 años. 
Después del infarto diagnosticaron insuficiencia cardíaca crónica, tratada con diurético.
Quirúrgicos: No se consignan cirugías de relevancia.
Alérgicos: Niega alergias conocidas.

5. Medicación habitual
Furosemida 40 mg/día; la paciente aumenta por cuenta propia a 80 mg/día algunos días cuando nota más edema.
Glibenclamida: 10 mg/día. 
Carvedilol 12,5 mg/día
Rosuvastatina 20 mg/día.
Losartan 50 mg/día.
Refiere adherencia irregular a las indicaciones dietarias y a los controles clínicos.

6. Hábitos
Niega tabaquismo, consumo de alcohol solo ocasionalmente.
Sedentarismo actual condicionado por la disnea.
Refiere ingesta de sal mayor a la recomendada y no realiza control diario de peso.

7. Antecedentes familiares
Antecedentes familiares de diabetes mellitus tipo 2 e hipertensión arterial en varios miembros de la familia. No se consignan otros antecedentes familiares de relevancia cardiovascular.
"""

physical_exam = """TA: 110/70 mmHg
FC: 90 lpm, iregular
FR: 30 rpm
Temperatura: 36,6 °C
SatO₂: 93 % AA
Peso y antropometría: peso habitual 67 kg; peso actual 74 kg; talla 1,68 m; IMC actual 26,2 kg/m².
Inspección general: paciente lúcida, orientada, disneica al hablar frases prolongadas, sin cianosis.
Cuello: ingurgitación yugular marcada que persiste en posición sentada, con escaso colapso inspiratorio.
Aparato cardiovascular: frecuencia 90 lpm, ritmo irregular. Latido apexiano desplazado al 6.º espacio intercostal izquierdo sobre línea axilar anterior, amplio y extenso. Soplo sistólico 2/6 en borde esternal inferior izquierdo, con ligero incremento durante la inspiración. Pulsos periféricos presentes.
Aparato respiratorio: murmullo vesicular globalmente disminuido. Rales crepitantes finos bibasales, de predominio inspiratorio tardío, que no se modifican con la tos.
Abdomen: blando, depresible; dolor a la palpación en epigastrio e hipocondrio derecho. Hígado palpable aproximadamente 3 traveses de dedo por debajo del reborde costal, doloroso, con altura hepática aproximada de 17 cm. Sin signos de irritación peritoneal.
Miembros inferiores: edema blando, bilateral, con fóvea, hasta ambas rodillas.
Sistema nervioso: sin focalidad neurológica evidente.
"""

extra_studies = """Electrocardiograma: Fibrilacion Auricular a 88 lpm. Ondas Q patológicas en derivaciones inferiores, compatibles con necrosis inferior antigua. Sin signos de isquemia aguda.

Radiografía de tórax: cardiomegalia, redistribución vascular hacia vértices, aumento de la trama intersticial bibasal y pequeños derrames pleurales bilaterales.

Laboratorio: hemoglobina 12,1 g/dL; leucocitos 7.600/mm³; plaquetas 228.000/mm³. Glucemia 178 mg/dL; HbA1c 8,2 %. Urea 48 mg/dL; creatinina 1,2 mg/dL; sodio 134 mEq/L; potasio 3,5 mEq/L. AST 42 U/L, ALT 39 U/L, bilirrubina total 1,1 mg/dL. NT-proBNP 2.450 pg/mL. Troponina ultrasensible sin ascenso dinámico.

Ecocardiograma Doppler: ventrículo izquierdo dilatado, con hipocinesia global y acinesia inferior. Fracción de eyección del ventrículo izquierdo aproximada de 32 %. Aurícula izquierda dilatada. Insuficiencia mitral funcional leve e insuficiencia tricuspídea moderada. Vena cava inferior dilatada con escaso colapso inspiratorio.
"""
