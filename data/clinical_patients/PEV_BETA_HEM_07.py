patient_prompt = """
GUION DE LA ENFERMEDAD (Solo revela esta información si la pregunta del estudiante lo requiere lógicamente)

1. Identificación y datos de filiación
Nombre y Apellido: Sabina Carrara.
Género: Femenino
Edad: 22 años
Lugar de residencia: San Luis
Ocupación: Estudiante de Fisioterapia
Nivel educativo: Superior / universitaria incompleta
Estado civil: Soltera

2. Motivo de consulta
Astenia, disnea y palpitaciones.

3. Anamnesis / Enfermedad actual
Paciente lúcida, de 22 años que refiere que hace aproximadamente 2-3 meses comenzó con astenia, disnea y palpitaciones al realizar esfuerzos moderados. Inicialmente atribuyó los síntomas a las altas temperaturas de los meses de enero y febrero y suspendió sus clases habituales de aerobic. En el último mes nota progresión: presenta disnea y palpitaciones con esfuerzos cotidianos, como subir un piso por escalera hasta su departamento, y refiere menor tolerancia al ejercicio y cansancio al finalizar las actividades habituales.
Niega dolor precordial, síncope, ortopnea, disnea paroxística nocturna, fiebre o síntomas respiratorios infecciosos. No refiere sangrados visibles, epistaxis, hematuria, hematemesis, melena ni hematoquecia. La catarsis es diaria, con heces formadas.
Desde hace más de un año sigue una alimentación vegana basada principalmente en frutas, verduras, semillas y harinas, sin consumo de carne, lácteos ni huevos. No realiza seguimiento nutricional regular ni suplementación sistemática con hierro. En los últimos meses perdió aproximadamente 3 kg. Refiere ciclos previamente regulares de 28/4 días, sin menstruaciones abundantes, aunque presentó oligomenorrea durante los últimos 3 meses.

4. Antecedentes personales
Fisiológicos y nutricionales: dieta vegana desde hace más de un año. Previamente realizaba actividad aeróbica con regularidad, suspendida por la aparición de los síntomas. Catarsis una vez por día, de características normales.
Gineco-obstétricos: menarca a los 13 años. Gesta 0. Ciclos habituales 28/4; oligomenorrea en los últimos 3 meses. Niega hipermenorrea, metrorragia o sangrado intermenstrual. Método anticonceptivo: preservativo. Pareja estable desde hace 8 meses.
Patológicos: sin enfermedades crónicas conocidas. Apendicectomía por apendicitis aguda a los 20 años. Niega antecedentes de enfermedad gastrointestinal, renal o hematológica.
Alergias: no refiere alergias medicamentosas conocidas.

5. Medicación habitual
No utiliza medicación habitual.
No recibe suplementos de hierro. No refiere automedicación habitual.

6. Hábitos
No fuma. Consume aproximadamente 2 vasos de cerveza durante los fines de semana. Niega consumo de drogas recreativas.
Actividad física previamente regular; suspendió las clases de aerobic por disnea y palpitaciones.

7. Antecedentes familiares
Madre viva, con hipotiroidismo en tratamiento. Padre vivo, hipertenso en tratamiento. Dos hermanos sanos.
Niega antecedentes familiares conocidos de anemia hereditaria, hemoglobinopatías o enfermedades hematológicas.
"""

physical_exam = """TA: 100/60 mmHg

FC: 80 lpm, regular

FR: 18 rpm

Temperatura: 36,6 °C

SatO₂: 99 % AA

Peso habitual: 54 kg. Peso actual: 51 kg. Talla: 1,62 m. IMC: 19,4 kg/m². Paciente lúcida, orientada y colaboradora. Se observa palidez marcada de piel y mucosas.

Tejido celular subcutáneo: sin adenomegalias palpables. Edema maleolar leve bilateral (+).

Aparato respiratorio: tórax simétrico, buena entrada bilateral de aire, murmullo vesicular conservado, sin ruidos agregados.

Aparato cardiovascular: ruidos cardíacos rítmicos. Soplo sistólico 2/6, de características eyectivas, audible en ápex y mesocardio. Pulsos periféricos presentes y simétricos. Sin ingurgitación yugular.

Abdomen: blando, depresible, indoloro. Sin hepatomegalia ni esplenomegalia.

Neurológico: sin déficit motor o sensitivo evidente.
"""

extra_studies = """Hemograma: leucocitos 4,4 × 10⁹/L; neutrófilos 61 %, eosinófilos 1 %, linfocitos 32 %, monocitos 6 %. Eritrocitos 3,57 × 10¹²/L; hematocrito 22,9 %; hemoglobina 7,1 g/dL; VCM 64 fL; HCM 18,3 pg; CHCM 28,6 g/dL; ADE/RDW 20,7 %. Plaquetas 294 × 10⁹/L.

Frotis de sangre periférica: marcada anisocitosis, abundantes microcitos, hipocromía moderada y poiquilocitosis.

Reticulocitos: 1,1 % (respuesta reticulocitaria inadecuada para el grado de anemia).

Metabolismo del hierro: ferremia 23 µg/dL (disminuida); transferrina 455 mg/dL (elevada); saturación de transferrina 6 % (disminuida); ferritina 5 ng/mL (disminuida).

Función renal, hepatograma y TSH: sin alteraciones relevantes.
"""