# FerroPINN

## 🎯 Overview and objectives
This repository contains Python codes that simulate the lid-driven cavity flow using Physics-Informed Neural Networks (PINNs). The main library employed is [PyTorch](https://pypi.org/project/torch/)
, which provides the core tools required to implement and train PINN models.

The primary objective of this project is to investigate the efficiency of PINNs, starting with a simple validation case (the lid-driven cavity) and subsequently advancing to a more complex physical problem: the thermoconvection of magnetic fluids under an applied magnetic field.

This repository is associated with the **Laboratório de Computação Científica em Escoamentos Complexos (LCEC-UNB)**.

## 📁 Estrutura de pastas e arquivos

O presente repositório possui a estrutura abaixo.

FerroPINN/
├── README.md
├── LICENSE
├── .gitignore
├── src/
├── docs/
└── examples/

- `src/` → códigos-fonte do projeto  
- `examples/` → casos de teste e exemplos de simulação  
- `docs/` → documentação, artigos e anotações técnicas 

## 📝 Codes

### cavidade_cisalhante.py

This script implements a baseline PINN solver for the two-dimensional lid-driven cavity flow, running a single simulation per execution. Training is performed using the Adam optimizer, with an optional switch to L-BFGS after 5000 epochs, making it suitable for validation studies and direct comparison with automated or optimized implementations.

### cavidade_cisalhante_sweep.py

This script extends the baseline PINN solver for the two-dimensional lid-driven cavity flow by enabling multiple simulations within a single execution. It allows the user to define a set of simulations with different physical parameters, neural network architectures, and loss weights, which are then executed sequentially.

The script is designed for parameter sweeps and comparative studies, automatically organizing the results of each simulation into structured directories according to the Reynolds number. Unlike the baseline version, this implementation prioritizes automation and reproducibility over interactivity, making it suitable for systematic numerical experiments.

# 🧭 Guia de Boas Práticas – Como escrever um bom README.md

Um bom `README.md` é o **cartão de visita do seu projeto científico**.  
Ele deve permitir que qualquer pessoa (inclusive você, no futuro!) entenda rapidamente  
**o que o código faz**, **como rodar**, **como contribuir** e **quais resultados esperar**.  

O arquivo README.md deve ser completo e descrever de maneira clara e interessante o que o programa faz, como faz, para que serve, qual o contexto de sua criação, artigos científicos vinculados ao programa e referências bibliográficas. O README.md pode conter imagens e equações científicas usando sintaxe LaTeX. Apenas garanta que essas equações fiquem visíveis ao subir o README.md para o GitHuB. As imagens ilustrativas contidas no README.md podem ser armazenadas dentro da pasta examples e de preferência em formato PNG. Para documentações muito extensas, você pode criar seções e um sumário no início do arquivo README.md (ver exemplos no repositório do simmsus: https://github.com/lcec-unb/simmsus). 

Abaixo estão as **boas práticas recomendadas pelo LCEC-UNB**.

---

## 📘 Estrutura mínima recomendada

```markdown
# Nome do Projeto
Breve descrição do objetivo e contexto científico do projeto.

## 🎯 Objetivo
Explique em 2–3 frases o que o programa resolve ou investiga.
Exemplo: “Simula o campo de temperatura em um tecido biológico sujeito a aquecimento magnético.”

## ⚙️ Estrutura de Pastas
Descreva como o projeto está organizado:
- `src/` – códigos-fonte principais
- `examples/` – casos de teste e exemplos de simulação
- `docs/` – relatórios, artigos, anotações e resultados
- `input/` (opcional) – arquivos de entrada
- `output/` (opcional) – resultados gerados

## 🚀 Execução
Explique como compilar e rodar:
```bash
make
./programa.exe < input.dat > output.log
```
Inclua também dependências (por exemplo, “necessita do compilador `gfortran` ou `ifx`”).

## 📊 Exemplos de Saída
Mostre exemplos reais: gráficos, tabelas ou prints de terminal.
Use figuras do diretório `examples/` ou `docs/`.

## 🧪 Metodologia / Modelos Matemáticos
Descreva brevemente o modelo físico ou matemático usado.
Se possível, cite referências bibliográficas (artigos, dissertações, teses).

## 👥 Autoria e Responsável
- **Autor principal:** André de Oliveira Brandão (2026)
- **Orientador:** Prof. Rafael Gabler Gontijo  
- **Laboratório:** [LCEC-UNB](https://github.com/LCEC-UNB)

## 📜 Licença
Indique a licença usada (ex: MIT).

## 📚 Referências
Liste até 3 referências que embasam o trabalho (artigos, livros, relatórios).

## 💡 Boas práticas
- Use Markdown com títulos (`#`, `##`, `###`), listas e blocos de código.  
- Prefira explicações curtas e claras.  
- Inclua imagens e figuras salvas em `docs/` usando:
  ```markdown
  ![Descrição da imagem](docs/figura1.png)
  ```
- Evite incluir arquivos binários pesados no repositório.  
- Atualize o `README.md` sempre que o projeto mudar.
---

## 🔧 Dicas extras

- Use emojis para deixar o README mais visual (🎯 ⚙️ 📊 💡).  
- Se o projeto gerar figuras de resultados, mostre **uma imagem de exemplo**.  
- Se houver publicação associada, inclua o DOI ou link para o artigo.  
- Atualize o README a cada etapa relevante (nova versão, caso de teste, release).

---

> 📘 **Este guia é parte do padrão oficial de documentação dos projetos do LCEC-UNB.**  
> Todos os novos repositórios devem incluir um README.md seguindo estas orientações.

## 👥 Contato
**Coordenador:** [Prof. Rafael Gabler Gontijo](http://www.rafaelgabler.com.br)   
**Organização:** [LCEC-UNB no GitHub](https://github.com/LCEC-UNB)
