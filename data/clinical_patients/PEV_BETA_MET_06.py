patient_prompt = """
GUION DE LA ENFERMEDAD (Solo revela esta información si la pregunta del estudiante lo requiere lógicamente)

1. Identificación y datos de filiación
Nombre y Apellido: Jorgelina Suárez
Género: Femenino
Edad: 48 años
Lugar de residencia: Mendiolaza, Córdoba
Ocupación: Secretaria en un estudio jurídico
Nivel educativo: Secundario completo
Estado civil: Casada
Otros datos de filiación relevantes: Vive con su esposo y uno de sus hijos

2. Motivo de consulta
Aumento progresivo de peso.

3. Anamnesis / Enfermedad actual
Paciente lucida de 48 años que consulta por aumento progresivo de aproximadamente 10 kg durante los últimos 12-18 meses. Su peso habitual era de 76-77 kg y actualmente pesa 87 kg. Refiere tendencia previa al sobrepeso, pero mayor dificultad para controlar el peso desde que aumentaron sus horas laborales. Permanece sentada 8-10 horas por día y redujo de forma marcada la actividad física.
Ha realizado dietas por cuenta propia, con descensos transitorios de 2-3 kg y posterior recuperación. Refiere mayor apetito por la tarde y noche, ingesta frecuente de panificados, galletitas y dulces durante el trabajo, consumo habitual de harinas y carnes, baja ingesta de frutas y verduras y bebidas azucaradas 2-3 veces por semana. Camina 20-30 minutos una o dos veces por semana, sin ejercicio programado.
En los últimos meses nota mayor cansancio al finalizar la jornada y menor tolerancia a esfuerzos como caminar en pendiente o subir varios pisos. Niega disnea de reposo, ortopnea, disnea paroxística nocturna y dolor precordial. Refiere cefalea ocasional. Niega poliuria, polidipsia, pérdida involuntaria de peso, temblor, sudoración, intolerancia marcada al frío o calor y cambios intestinales persistentes.
Su esposo refiere ronquidos nocturnos. La paciente reconoce sueño poco reparador algunos días y somnolencia ocasional después del almuerzo, sin episodios de sueño involuntario durante el trabajo o al conducir. Consulta preocupada por el aumento de peso y por sus antecedentes familiares de diabetes.


4. Antecedentes personales
Fisiológicos y estilo de vida: alimentación mixta, con elevado consumo de hidratos de carbono refinados y alimentos ultraprocesados. Sedentarismo relacionado con actividad laboral. Estrés laboral frecuente. Duerme 6-7 horas por noche.
Patológicos: hipertensión arterial diagnosticada hace aproximadamente 6 años, en tratamiento con losartán 50 mg/día, sin controles domiciliarios regulares. Hipotiroidismo diagnosticado hace 8 años, tratado con levotiroxina 100 µg/día, con buena adherencia referida. Esteatosis hepática diagnosticada por ecografía hace 2 años. Niega diagnóstico previo de diabetes mellitus tipo 2. Refiere antecedente de triglicéridos elevados en controles previos.
Gineco-obstétricos: menopausia a los 43 años, sin terapia hormonal de reemplazo. Tres embarazos, tres partos, sin abortos. Antecedente de diabetes gestacional en el último embarazo, controlada con plan alimentario, sin requerimiento de insulina.
Quirúrgicos: colecistectomía laparoscópica a los 39 años por litiasis vesicular. Alergias: no refiere alergias medicamentosas conocidas.

5. Medicación habitual
Losartán 50 mg/día.
Levotiroxina 100 µg/día.
Ácido acetilsalicílico 100 mg/día, indicado años atrás durante un control cardiológico; desconoce el motivo preciso de la indicación.
Analgésicos ocasionales por cefalea. Niega tratamiento con hipoglucemiantes o hipolipemiantes.

6. Hábitos
No fuma. Consume alcohol ocasionalmente, principalmente fines de semana: aproximadamente 1-2 copas de vino. Niega drogas recreativas. Se moviliza habitualmente en automóvil. Actividad física escasa e irregular.

7. Antecedentes familiares
Madre con diabetes mellitus tipo 2 e hipertensión arterial. Padre con obesidad, fallecido a los 72 años por infarto agudo de miocardio. Tres hermanos; uno presenta obesidad severa y fue sometido a cirugía bariátrica. Niega antecedentes familiares conocidos de enfermedad tiroidea autoinmune.
"""

physical_exam = """TA: 145/80 mmHg

FC: 75 lpm, regular

FR: 16 rpm

Temperatura: 36,4 °C

SatO₂: 98 % AA

Peso: 87 kg. Talla: 1,65 m. IMC: 32,0 kg/m². Circunferencia de cintura: 104 cm. Predominio de adiposidad abdominal. Paciente lúcida, orientada y colaboradora, en buen estado general.

Piel y faneras: acantosis nigricans en región cervical posterior y axilas. Sin xantomas. Várices superficiales bilaterales de predominio distal.

Cabeza y cuello: mucosas húmedas y normocoloreadas. Tiroides no aumentada de tamaño, sin nódulos palpables. Sin adenopatías.

Cardiovascular: ruidos cardíacos rítmicos, sin soplos. Pulsos periféricos presentes y simétricos. Sin ingurgitación yugular ni edemas.

Respiratorio: buena entrada bilateral de aire, murmullo vesicular conservado, sin ruidos agregados.

Abdomen: globuloso a predominio adiposo, blando, depresible e indoloro. Sin visceromegalias palpables. Ruidos hidroaéreos conservados.

Neurológico: lúcida y orientada, fuerza y sensibilidad conservadas, sin focalidad.
"""

extra_studies = """Hemograma: Hb 13,7 g/dL; Hto 41 %; leucocitos 7.200/mm³; plaquetas 278.000/mm³.

Metabolismo glucídico: glucemia en ayunas 116 mg/dL; HbA1c 6,1 %.

Perfil lipídico: colesterol total 226 mg/dL; triglicéridos 238 mg/dL; HDL-c 38 mg/dL; LDL-c 140 mg/dL.

Función renal: urea 31 mg/dL; creatinina 0,78 mg/dL. Orina completa sin proteinuria ni glucosuria. Relación albúmina/creatinina urinaria dentro de límites normales.

Hepatograma: AST 37 U/L; ALT 52 U/L; FA 94 U/L; GGT 48 U/L; bilirrubina total 0,7 mg/dL; albúmina 4,3 g/dL.

Función tiroidea: TSH 2,4 mUI/L; T4 libre dentro de límites normales.

ECG: ritmo sinusal, FC 72 lpm, sin alteraciones significativas del segmento ST-T.

Ecografía abdominal: hígado discretamente aumentado de ecogenicidad en forma difusa, compatible con esteatosis hepática; sin lesiones focales. Vía biliar no dilatada. Bazo de tamaño normal. Colecistectomizada.
"""
