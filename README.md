# Proyecto ISII 2025-26 - Repositorio de Código

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

## 🏷️ Convención de Tags

### Formato

`vX.Y.Z-CICLO`

### Ejemplos

- `v1.0.0-C1`: Entrega final Ciclo 1
- `v1.1.0-C1`: Correcciones post-entrega Ciclo 1
- `v2.0.0-C2`: Entrega final Ciclo 2

### Tags de Hitos

- `milestone-diseno-c1`: Finalización fase diseño Ciclo 1
- `milestone-pruebas-c1`: Finalización fase pruebas Ciclo 1
- `milestone-diseno-c2`: Finalización fase diseño Ciclo 2
- `milestone-pruebas-c2`: Finalización fase pruebas Ciclo 2

## 💬 Convención de Commits

### Formato

<tipo>(<alcance>): <descripción>
<cuerpo opcional>
<pie opcional>

```
Tipos de Commit

feat: Nueva funcionalidad
fix: Corrección de errores
docs: Cambios en documentación
style: Cambios de formato, espacios, etc.
refactor: Refactorización de código
test: Añadir o modificar tests
chore: Tareas de mantenimiento

Alcances Sugeridos

c1: Cambios específicos del Ciclo 1
c2: Cambios específicos del Ciclo 2
config: Configuración del proyecto
db: Base de datos
ui: Interfaz de usuario

Ejemplos
feat(c1): implementar login de usuarios
fix(c1): corregir validación de email en registro
docs: actualizar README con instrucciones de instalación
test(c2): añadir tests unitarios para módulo de reportes
👥 Configuración del Equipo
Configuración Inicial de Git
Cada miembro debe ejecutar:
bashgit config --global user.name "Tu Nombre Completo"
git config --global user.email "tu.email@alumnos.upm.es"
git config --global init.defaultBranch main
Roles y Responsabilidades

Líder del Proyecto: Gestiona main y release branches
Responsable de Desarrollo: Gestiona develop branches
Responsable de GCS: Gestiona tags y merges críticos
Responsable de Calidad: Revisa PRs antes de merge

🔄 Flujo de Trabajo
Para Funcionalidades Nuevas

Crear branch desde cicloX/develop
Desarrollar funcionalidad
Hacer commits siguiendo convenciones
Push y crear Pull Request
Revisión de código
Merge a cicloX/develop

Para Entregas

Crear release/vX.Y.Z-CX desde cicloX/develop
Testing final y correcciones
Merge a main
Tag de versión
Merge de vuelta a develop

🚀 Comandos Útiles
Trabajo Diario
bash# Cambiar a rama de trabajo
git checkout ciclo1/develop

# Crear nueva funcionalidad
git checkout -b ciclo1/feature/nombre-funcionalidad

# Ver estado
git status

# Añadir cambios
git add .

# Commit con mensaje apropiado
git commit -m "feat(c1): descripción del cambio"

# Subir cambios
git push origin ciclo1/feature/nombre-funcionalidad
Gestión de Ramas
bash# Listar todas las ramas
git branch -a

# Sincronizar con remoto
git fetch --all

# Merge desde develop
git merge ciclo1/develop
📊 Integración con Google Drive

Documentación: Google Drive
Código fuente: Git
Binarios/Ejecutables: Git (tags de release)
Recursos gráficos: Google Drive con referencias en código

⚠️ Reglas Importantes

NUNCA hacer push directo a main
SIEMPRE crear Pull Request para cambios importantes
OBLIGATORIO mensaje de commit descriptivo
REQUERIDO revisión de código antes de merge
NECESARIO tag para cada entrega oficial

Documento actualizado: 20 Septiembre 2025
```
