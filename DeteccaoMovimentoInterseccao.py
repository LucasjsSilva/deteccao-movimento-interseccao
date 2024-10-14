import shutil

import cv2
import os

nomeVideo = 'meu_video2'
videoPath = f'{nomeVideo}.mp4'
outputVideoPath = f'video_movimento_intersecao_{nomeVideo}.avi'
outputFramesPath = f'frames_com_movimento_{nomeVideo}'  # Pasta para salvar os frames detectados
movementThreshold = 0.01  # Ajuste o limite conforme necessário

# Cria a pasta de saída para os frames, se não existir
if os.path.exists(outputFramesPath):
    shutil.rmtree(outputFramesPath)

os.makedirs(outputFramesPath)

cap = cv2.VideoCapture(videoPath)
ret, firstFrame = cap.read()

if not ret:
    print("Erro ao ler o vídeo.")
    cap.release()
    exit()

# Aplica o filtro Gaussiano para suavizar a imagem
grayFirstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
grayFirstFrame = cv2.GaussianBlur(grayFirstFrame, (5, 5), 0)

height, width = grayFirstFrame.shape

gridSizeWidth = width//10  # Grade 128x128
gridSizeHeight = height//10

cellHeight, cellWidth = height // gridSizeHeight, width // gridSizeWidth

# Prepara o vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outputVideo = cv2.VideoWriter(outputVideoPath, fourcc, 20.0, (width, height))

frameCount = 0  # Contador de frames

while True:
    ret, secondFrame = cap.read()
    if not ret:
        break

    graySecondFrame = cv2.cvtColor(secondFrame, cv2.COLOR_BGR2GRAY)
    graySecondFrame = cv2.GaussianBlur(graySecondFrame, (5, 5), 0)

    # Criar uma cópia do frame original para sobrepor o movimento
    frameWithMovement = secondFrame.copy()
    movementDetected = False  # Variável para verificar se houve movimento no frame

    for i in range(int(gridSizeHeight)):
        for j in range(int(gridSizeWidth)):
            xStart = j * cellWidth
            xEnd = (j + 1) * cellWidth
            yStart = i * cellHeight
            yEnd = (i + 1) * cellHeight

            firstFrameCell = grayFirstFrame[yStart:yEnd, xStart:xEnd]
            secondFrameCell = graySecondFrame[yStart:yEnd, xStart:xEnd]

            # Cálculo dos histogramas
            histFirst = cv2.calcHist([firstFrameCell], [0], None, [256], [0, 256])
            histSecond = cv2.calcHist([secondFrameCell], [0], None, [256], [0, 256])

            # Normaliza os histogramas
            cv2.normalize(histFirst, histFirst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(histSecond, histSecond, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Compara os histogramas usando a interseção
            correlation = cv2.compareHist(histFirst, histSecond, cv2.HISTCMP_INTERSECT)

            # Se a correlação for baixa, significa que houve movimento
            if correlation < movementThreshold:
                movementDetected = True
                # Preenche as áreas com movimento com cor vermelha sólida
                cv2.rectangle(frameWithMovement, (xStart, yStart), (xEnd, yEnd), (0, 0, 255), cv2.FILLED)

    # Escreve o frame no vídeo de saída
    outputVideo.write(frameWithMovement)

    # Se houve movimento, salva o frame na pasta
    frameFilename = f"{outputFramesPath}/frame_{frameCount:04d}.png"
    cv2.imwrite(frameFilename, frameWithMovement)

    # Atualiza o frame anterior
    grayFirstFrame = graySecondFrame.copy()
    frameCount += 1

cap.release()
outputVideo.release()
