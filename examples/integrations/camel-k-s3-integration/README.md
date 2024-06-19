https://access.redhat.com/documentation/en-us/red_hat_build_of_apache_camel/4.4/html/getting_started_with_red_hat_build_of_apache_camel_for_quarkus/getting-started-with-camel-quarkus-extensions_camel-quarkus-extensions#generating_the_skeleton_application_with_code_quarkus_redhat_com


https://access.redhat.com/documentation/en-us/red_hat_build_of_apache_camel/4.4/html/getting_started_with_red_hat_build_of_apache_camel_for_quarkus/set-up-maven-locally

```sh
mvn io.quarkus:quarkus-maven-plugin:create \
    -DprojectGroupId=com.redhat.docbot \
    -DprojectArtifactId=camel-k-s3-integration \
    -DclassName="com.redhat.docbot.S3EventListener" \
    -Dpath="/hello"
cd camel-k-s3-integration
```

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.apache.camel.quarkus</groupId>
            <artifactId>camel-quarkus-bom</artifactId>
            <version>${camel-quarkus.version}</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<dependencies>
    <dependency>
        <groupId>org.apache.camel.quarkus</groupId>
        <artifactId>camel-quarkus-aws2-s3</artifactId>
    </dependency>
    <dependency>
        <groupId>org.apache.camel.k</groupId>
        <artifactId>camel-k-runtime</artifactId>
    </dependency>
</dependencies>
```

# camel-k-s3-integration

This project uses Quarkus, the Supersonic Subatomic Java Framework.

If you want to learn more about Quarkus, please visit its website: https://quarkus.io/ .

## Running the application in dev mode

You can run your application in dev mode that enables live coding using:
```shell script
./mvnw compile quarkus:dev
```

> **_NOTE:_**  Quarkus now ships with a Dev UI, which is available in dev mode only at http://localhost:8080/q/dev/.

## Packaging and running the application

The application can be packaged using:
```shell script
./mvnw package
```
It produces the `quarkus-run.jar` file in the `target/quarkus-app/` directory.
Be aware that it’s not an _über-jar_ as the dependencies are copied into the `target/quarkus-app/lib/` directory.

The application is now runnable using `java -jar target/quarkus-app/quarkus-run.jar`.

If you want to build an _über-jar_, execute the following command:
```shell script
./mvnw package -Dquarkus.package.type=uber-jar
```

The application, packaged as an _über-jar_, is now runnable using `java -jar target/*-runner.jar`.

## Creating a native executable

You can create a native executable using: 
```shell script
./mvnw package -Dnative
```

Or, if you don't have GraalVM installed, you can run the native executable build in a container using: 
```shell script
./mvnw package -Dnative -Dquarkus.native.container-build=true
```

You can then execute your native executable with: `./target/camel-k-s3-integration-1.0.0-SNAPSHOT-runner`

If you want to learn more about building native executables, please consult https://quarkus.io/guides/maven-tooling.

## Related Guides

- Camel Core ([guide](https://access.redhat.com/documentation/en-us/red_hat_build_of_apache_camel/4.4/html/red_hat_build_of_apache_camel_for_quarkus_reference/camel-quarkus-extensions-reference#extensions-core)): Camel core functionality and basic Camel languages: Constant, ExchangeProperty, Header, Ref, Simple and Tokenize
- Camel AWS 2 S3 Storage Service ([guide](https://access.redhat.com/documentation/en-us/red_hat_build_of_apache_camel/4.4/html/red_hat_build_of_apache_camel_for_quarkus_reference/camel-quarkus-extensions-reference#extensions-aws2-s3)): Store and retrieve objects from AWS S3 Storage Service
