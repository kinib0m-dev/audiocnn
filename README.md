# Proyecto ISII - Repositorio de Código

## Estrategia de Branching

### Ramas Principales

- `main`: Rama principal con código estable y probado
- `develop`: Rama de desarrollo donde se integran las funcionalidades

### Ramas por Ciclo

- `ciclo1/develop`: Desarrollo del Ciclo 1
- `ciclo1/feature/*`: Funcionalidades específicas del Ciclo 1
- `ciclo2/develop`: Desarrollo del Ciclo 2
- `ciclo2/feature/*`: Funcionalidades específicas del Ciclo 2

### Ramas de Soporte

- `hotfix/*`: Correcciones urgentes en main
- `release/*`: Preparación de entregas

## Convención de Tags

### Formato

`vX.Y.Z-CICLO`

### Ejemplos

- `v1.0.0-C1`: Entrega final Ciclo 1
- `v1.1.0-C1`: Correcciones post-entrega Ciclo 1
- `v2.0.0-C2`: Entrega final Ciclo 2

## Convención de Commits

### Formato

Tipos de Commit

`feat`: Nueva funcionalidad
`fix`: Corrección de errores
`docs`: Cambios en documentación
`style`: Cambios de formato, espacios, etc.
`refactor`: Refactorización de código
`test`: Añadir o modificar tests
`chore`: Tareas de mantenimiento

### Ejemplos

feat(c1): implementar login de usuarios
fix(c1): corregir validación de email en registro
docs: actualizar README con instrucciones de instalación
test(c2): añadir tests unitarios para módulo de reportes

## Convención de Commits

### Configuración Inicial de Git

Cada miembro debe ejecutar:
git config --global user.name "Nombre"
git config --global init.defaultBranch main

## Flujo de Trabajo

### Para Funcionalidades Nuevas

Crear branch desde cicloX/develop
Desarrollar funcionalidad
Hacer commits siguiendo convenciones
Push y crear Pull Request
Revisión de código
Merge a cicloX/develop

### Para Entregas

Crear release/vX.Y.Z-CX desde cicloX/develop
Testing final y correcciones
Merge a main
Tag de versión
Merge de vuelta a develop

## Comandos Útiles

# Cambiar a rama de trabajo

`git checkout ciclo1/develop`

# Crear nueva funcionalidad

`git checkout -b ciclo1/feature/nombre-funcionalidad`

# Ver estado

`git status`

# Añadir cambios

`git add .`

# Commit con mensaje apropiado

`git commit -m "feat(c1): descripción del cambio"`

# Subir cambios

`git push origin ciclo1/feature/nombre-funcionalidad`

Documento actualizado: 20 Septiembre 2025
