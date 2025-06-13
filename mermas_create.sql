-- Created by Vertabelo (http://vertabelo.com)
-- Last modification date: 2025-05-14 19:28:56.744

-- tables
-- Table: dim_Linea
CREATE TABLE dim_Linea (
    id_linea bigint  NOT NULL,
    nombre_linea varchar  NOT NULL,
    categoria_linea varchar  NOT NULL,
    abastecimiento_linea varchar  NOT NULL,
    CONSTRAINT dim_Linea_pk PRIMARY KEY (id_linea)
);

-- Table: dim_Motivo
CREATE TABLE dim_Motivo (
    id_motivo bigint  NOT NULL,
    nombre_motivo varchar  NOT NULL,
    ubicacion_motivo varchar  NOT NULL,
    CONSTRAINT dim_Motivo_pk PRIMARY KEY (id_motivo)
);

-- Table: dim_Negocio
CREATE TABLE dim_Negocio (
    id_negocio bigint  NOT NULL,
    nombre_negocio varchar  NOT NULL,
    categoria_negocio varchar  NOT NULL,
    subcategoria_negocio varchar  NOT NULL,
    CONSTRAINT dim_Negocio_pk PRIMARY KEY (id_negocio)
);

-- Table: dim_Producto
CREATE TABLE dim_Producto (
    id_producto bigint  NOT NULL,
    descripcion varchar  NOT NULL,
    categoria varchar  NOT NULL,
    subcategoria varchar  NOT NULL,
    abastecimiento varchar  NOT NULL,
    marca varchar  NOT NULL,
    origen varchar  NOT NULL,
    CONSTRAINT dim_Producto_pk PRIMARY KEY (id_producto)
);

-- Table: dim_Seccion
CREATE TABLE dim_Seccion (
    id_seccion bigint  NOT NULL,
    nombre_seccion varchar  NOT NULL,
    tipo_ambiente varchar  NOT NULL,
    visibilidad varchar  NOT NULL,
    nivel_acceso_cliente varchar  NOT NULL,
    CONSTRAINT dim_Seccion_pk PRIMARY KEY (id_seccion)
);

-- Table: dim_Tiempo
CREATE TABLE dim_Tiempo (
    id_tiempo bigint  NOT NULL,
    nombre_dia varchar  NOT NULL,
    nombre_mes varchar  NOT NULL,
    ano int  NOT NULL,
    semestre int  NOT NULL,
    trimestre int  NOT NULL,
    feriado boolean  NOT NULL,
    fecha date  NOT NULL,
    estacion varchar  NOT NULL,
    nombre_semestre varchar  NOT NULL,
    nombre_trimestre varchar(256)  NOT NULL,
    numero_dia int  NOT NULL,
    numero_mes int  NOT NULL,
    numero_semana int  NOT NULL,
    fin_de_semana boolean  NOT NULL,
    CONSTRAINT dim_Tiempo_pk PRIMARY KEY (id_tiempo)
);

-- Table: dim_Tienda
CREATE TABLE dim_Tienda (
    id_tienda bigint  NOT NULL,
    nombre_tienda varchar  NOT NULL,
    region varchar  NOT NULL,
    comuna varchar  NOT NULL,
    zonal varchar  NOT NULL,
    tipo_tienda varchar  NOT NULL,
    CONSTRAINT dim_Tienda_pk PRIMARY KEY (id_tienda)
);

-- Table: fact_Mermas
CREATE TABLE fact_Mermas (
    id_merma bigint  NOT NULL,
    merma_unidad int  NOT NULL,
    merma_monto int  NOT NULL,
    dim_Motivo_id_motivo bigint  NOT NULL,
    dim_Tiempo_id_tiempo bigint  NOT NULL,
    dim_Tienda_id_tienda bigint  NOT NULL,
    dim_Producto_id_producto bigint  NOT NULL,
    dim_Seccion_id_seccion bigint  NOT NULL,
    dim_Negocio_id_negocio bigint  NOT NULL,
    dim_Linea_id_linea bigint  NOT NULL,
    CONSTRAINT fact_Mermas_pk PRIMARY KEY (id_merma)
);

-- foreign keys
-- Reference: fact_Mermas_dim_Linea (table: fact_Mermas)
ALTER TABLE fact_Mermas ADD CONSTRAINT fact_Mermas_dim_Linea
    FOREIGN KEY (dim_Linea_id_linea)
    REFERENCES dim_Linea (id_linea)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: fact_Mermas_dim_Motivo (table: fact_Mermas)
ALTER TABLE fact_Mermas ADD CONSTRAINT fact_Mermas_dim_Motivo
    FOREIGN KEY (dim_Motivo_id_motivo)
    REFERENCES dim_Motivo (id_motivo)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: fact_Mermas_dim_Negocio (table: fact_Mermas)
ALTER TABLE fact_Mermas ADD CONSTRAINT fact_Mermas_dim_Negocio
    FOREIGN KEY (dim_Negocio_id_negocio)
    REFERENCES dim_Negocio (id_negocio)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: fact_Mermas_dim_Producto (table: fact_Mermas)
ALTER TABLE fact_Mermas ADD CONSTRAINT fact_Mermas_dim_Producto
    FOREIGN KEY (dim_Producto_id_producto)
    REFERENCES dim_Producto (id_producto)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: fact_Mermas_dim_Seccion (table: fact_Mermas)
ALTER TABLE fact_Mermas ADD CONSTRAINT fact_Mermas_dim_Seccion
    FOREIGN KEY (dim_Seccion_id_seccion)
    REFERENCES dim_Seccion (id_seccion)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: fact_Mermas_dim_Tiempo (table: fact_Mermas)
ALTER TABLE fact_Mermas ADD CONSTRAINT fact_Mermas_dim_Tiempo
    FOREIGN KEY (dim_Tiempo_id_tiempo)
    REFERENCES dim_Tiempo (id_tiempo)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: fact_Mermas_dim_Tienda (table: fact_Mermas)
ALTER TABLE fact_Mermas ADD CONSTRAINT fact_Mermas_dim_Tienda
    FOREIGN KEY (dim_Tienda_id_tienda)
    REFERENCES dim_Tienda (id_tienda)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- End of file.

